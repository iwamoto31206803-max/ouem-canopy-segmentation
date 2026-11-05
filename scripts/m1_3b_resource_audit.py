"""Opt-in CPU resource audit for one verified real Train patch."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import time

from m1_1_satlas_smoke import checkpoint_digest, gradient_evidence, validate_output
from m1_2d_geometry_audit import sha
from m1_3a_loader import load_patch


def parameter_summary(module):
    parameters = list(module.parameters())
    return dict(total=sum(p.numel() for p in parameters),
                trainable=sum(p.numel() for p in parameters if p.requires_grad),
                parameter_bytes=sum(p.numel()*p.element_size() for p in parameters),
                gradient_capacity_bytes=sum(p.numel()*p.element_size() for p in parameters if p.requires_grad))


def adam_estimate(trainable, parameter_bytes, gradient_bytes):
    state = trainable * 2 * 4
    return dict(unit='bytes',fp32_adam_moments_bytes=state,
                parameters_gradients_moments_bytes=parameter_bytes+gradient_bytes+state,
                exclusions='activations, buffers, workspaces, temporary optimizer tensors, allocator and runtime')


def backend(torch):
    return dict(torch_version=torch.__version__,cuda_available=torch.cuda.is_available(),
                cuda_build=torch.version.cuda,hip_build=torch.version.hip,
                cuda_device_count=torch.cuda.device_count(),gpu_vram_measurements='NOT_MEASURED')


def memory():
    if platform.system() != 'Windows':
        return dict(status='NOT_MEASURED',reason='Windows process counters required')
    class Counters(ctypes.Structure):
        _fields_ = [('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)] + [
            (name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize',
            'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    psapi = ctypes.WinDLL('psapi',use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(working_set_bytes=counters.WorkingSetSize,peak_working_set_bytes=counters.PeakWorkingSetSize,
                committed_private_bytes=counters.PagefileUsage,peak_committed_private_bytes=counters.PeakPagefileUsage)


def audit(data, checkpoint, record):
    import torch
    import satlaspretrain_models as satlas
    torch.set_num_threads(4)
    torch.manual_seed(0)
    record.update(python=platform.python_version(),platform=platform.platform(),threads=4,seed=0,
                  versions={p:version(p) for p in ['torch','torchvision','satlaspretrain-models']},
                  backend=backend(torch),memory_method='Windows GetProcessMemoryInfo, bytes, process-lifetime peak')
    command = 'Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,AdapterRAM | ConvertTo-Json -Compress'
    hardware = subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,text=True,check=True)
    record['video_controllers'] = json.loads(hardware.stdout)
    pairs = json.loads(Path('docs/implementation/M1_2D_PAIR_RESULT.json').read_text())
    if len({s['oam_id'] for s in pairs['samples']}) != len(pairs['samples']):
        raise ValueError('Source-group overlap')
    sample = next(s for s in pairs['samples'] if s['image_id']==661)
    images,targets,_ = load_patch(data,sample,'train',256)
    record['input'] = dict(image_id=661,split='train',crop=[0,0,256,256],
                         hashes=sample['output_hashes'],pair_manifest_sha256=sha(Path('docs/implementation/M1_2D_PAIR_RESULT.json')))
    record['checkpoint'] = json.loads(Path('docs/implementation/M1_1_PRETRAINED_CPU_RESULT.json').read_text())['checkpoint']
    checkpoint_digest(checkpoint,record['checkpoint']['sha256'])
    record['memory'] = {'before_load':memory()}
    times = {}
    started = time.perf_counter()
    weights = torch.load(checkpoint,map_location='cpu',weights_only=True)
    record['memory']['checkpoint_loaded'] = memory()
    model = satlas.Model(num_channels=3,multi_image=False,backbone=satlas.Backbone.SWINB,
                         fpn=True,head=satlas.Head.SEGMENT,num_categories=2,weights=weights).cpu()
    del weights
    model.requires_grad_(True)
    model.train()
    times['checkpoint_load_and_construction_seconds'] = time.perf_counter()-started
    record['memory']['model_ready'] = memory()
    record['parameters'] = {name:parameter_summary(getattr(model,name)) for name in ['backbone','fpn','upsample','head']}
    summary = parameter_summary(model)
    record['parameters']['whole_model'] = summary
    record['fp32_all_layer_estimate'] = adam_estimate(summary['trainable'],summary['parameter_bytes'],summary['gradient_capacity_bytes'])
    backbone = record['parameters']['backbone']
    record['fp32_frozen_backbone_estimate'] = adam_estimate(summary['trainable']-backbone['trainable'],
        summary['parameter_bytes'],summary['gradient_capacity_bytes']-backbone['gradient_capacity_bytes'])
    started = time.perf_counter()
    probabilities,loss = model(images,targets)
    times['forward_including_official_ce_seconds'] = time.perf_counter()-started
    record['memory']['after_forward_loss'] = memory()
    validate_output(torch,probabilities,loss,256)
    record['loss'] = loss.item()
    started = time.perf_counter()
    loss.backward()
    times['backward_seconds'] = time.perf_counter()-started
    record['memory']['after_backward'] = memory()
    record['gradients'] = {name:gradient_evidence(torch,getattr(model,name)) for name in ['backbone','fpn','upsample','head']}
    record['actual_gradient_tensor_bytes'] = sum(p.grad.numel()*p.grad.element_size() for p in model.parameters() if p.grad is not None)
    record['times'] = times
    record['status'] = 'TRAINING_RESOURCE_AUDIT_PASS_CANDIDATE'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    record=dict(status='NOT_VERIFIED',utc=datetime.now(timezone.utc).isoformat(),device='cpu',batch_size=1)
    started=time.perf_counter()
    code=0
    try:
        audit(args.data,args.checkpoint,record)
    except Exception as error:
        record.update(status='FAIL',error=f'{type(error).__name__}: {error}')
        code=1
    record['total_seconds']=time.perf_counter()-started
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
    raise SystemExit(code)
