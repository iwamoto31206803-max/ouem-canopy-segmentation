from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from m1_3b_resource_audit import parameter_summary, adam_estimate, backend, memory


def test_parameter_and_adam_units():
    layer=torch.nn.Linear(3,2)
    summary=parameter_summary(layer)
    assert summary==dict(total=8,trainable=8,parameter_bytes=32,gradient_capacity_bytes=32)
    estimate=adam_estimate(8,32,32)
    assert estimate['fp32_adam_moments_bytes']==64
    assert estimate['parameters_gradients_moments_bytes']==128
    layer.weight.requires_grad_(False)
    assert parameter_summary(layer)['trainable']==2
    assert parameter_summary(layer)['gradient_capacity_bytes']==8


def test_backend_without_gpu():
    fake=SimpleNamespace(__version__='test+cpu',cuda=SimpleNamespace(is_available=lambda:False,device_count=lambda:0),
                         version=SimpleNamespace(cuda=None,hip=None))
    assert backend(fake)['cuda_available'] is False
    assert backend(fake)['gpu_vram_measurements']=='NOT_MEASURED'


def test_memory_counters_or_explicit_unmeasured():
    result=memory()
    if result.get('status')=='NOT_MEASURED':
        assert result['reason']
    else:
        assert 0 < result['working_set_bytes'] <= result['peak_working_set_bytes']
        assert result['committed_private_bytes'] <= result['peak_committed_private_bytes']
