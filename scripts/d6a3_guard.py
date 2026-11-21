"""Windows guarded worker for synthetic D6 product staging."""
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import subprocess
import sys
import time

from r1d_bounded_prepare import WindowsJob
from research_checkpoint_audit import available_memory


class Memory(ctypes.Structure):
    _fields_ = [('size', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in ('peak', 'current', 'paged_peak', 'paged',
                                           'nonpaged_peak', 'nonpaged', 'commit', 'commit_peak')]


def watch(args, limits):
    if sys.platform != 'win32' or available_memory() < limits['host_available_bytes']:
        raise RuntimeError('RESOURCE_STOP: Windows enforcement or host RAM unavailable')
    read = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
    read.argtypes = [wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD]
    read.restype = wintypes.BOOL
    job = WindowsJob(limits['commit_bytes'])
    process = None
    start = time.monotonic()
    peak = 0
    try:
        bootstrap = (
            "import sys; assert sys.stdin.buffer.read(1)==b'1'; "
            "sys.path[:0]=sys.argv[1:3]; "
            "from d6a3_produce import worker; worker(*sys.argv[3:])"
        )
        site = Path(sys.prefix) / 'Lib' / 'site-packages'
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   GDAL_NUM_THREADS='1', GDAL_CACHEMAX='32', GDAL_PAM_ENABLED='NO', PROJ_NETWORK='OFF')
        process = subprocess.Popen(
            [sys._base_executable, '-I', '-B', '-S', '-c', bootstrap, str(site),
             str(Path(__file__).parent), *map(str, args)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=env, creationflags=subprocess.CREATE_NO_WINDOW)
        job.assign(process)
        process.stdin.write(b'1')
        process.stdin.close()
        while True:
            code = process.poll()
            counters = Memory()
            counters.size = ctypes.sizeof(counters)
            if not read(process._handle, ctypes.byref(counters), counters.size) or counters.peak <= 0:
                raise RuntimeError('RESOURCE_STOP: peak working set unavailable')
            peak = max(peak, counters.peak)
            elapsed = time.monotonic() - start
            if peak >= limits['rss_bytes'] or elapsed >= limits['seconds']:
                raise RuntimeError('RESOURCE_STOP: peak working set or deadline exceeded')
            if code is not None:
                if code:
                    raise RuntimeError('WORKER_FAILED: incomplete stage retained')
                return dict(elapsed_seconds=elapsed, peak_rss_bytes=peak,
                            peak_commit_bytes=job.query().peak_job)
            time.sleep(.025)
    finally:
        job.close()
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait()
