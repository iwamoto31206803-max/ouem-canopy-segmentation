"""Windows Job supervision for D6 staging or a complete pilot attempt."""
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


def watch(args, limits, *, attempt=False, qa_attempt=False):
    if sys.platform != 'win32' or available_memory() < limits['host_available_bytes']:
        raise RuntimeError('RESOURCE_STOP: Windows enforcement or host RAM unavailable')
    if qa_attempt and not attempt:
        raise ValueError('QA requires a whole-attempt deadline')
    deadline = limits['attempt_seconds' if attempt else 'seconds']
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
            + ("from d6a4 import qa_worker; qa_worker(*sys.argv[3:])" if qa_attempt
               else "from d6a32_gate import run_guarded; run_guarded(*sys.argv[3:])" if attempt
               else "from d6a3_produce import worker; worker(*sys.argv[3:])")
        )
        site = Path(sys.prefix) / 'Lib' / 'site-packages'
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   GDAL_NUM_THREADS='1', GDAL_CACHEMAX='32', GDAL_PAM_ENABLED='NO', PROJ_NETWORK='OFF')
        process = subprocess.Popen(
            [sys._base_executable, '-I', '-B', '-S', '-c', bootstrap, str(site),
             str(Path(__file__).parent), *map(str, args)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE if attempt else subprocess.DEVNULL,
            stderr=None if attempt else subprocess.DEVNULL,
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
            if peak >= limits['rss_bytes'] or elapsed >= deadline:
                raise RuntimeError('RESOURCE_STOP: peak working set or deadline exceeded')
            if code is not None:
                if attempt and code == 124:
                    raise RuntimeError('RESOURCE_STOP: source deadline exceeded')
                if code:
                    raise RuntimeError('WORKER_FAILED: inspect private output for any retained stage')
                stats = dict(elapsed_seconds=elapsed, peak_rss_bytes=peak,
                             peak_commit_bytes=job.query().peak_job)
                if attempt:
                    from d6a_research_qa import strict_json
                    return dict(strict_json(process.stdout.read(4096)), attempt_measurements=stats)
                return stats
            time.sleep(.025)
    finally:
        job.close()
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            if process.stdout is not None:
                process.stdout.close()


def allocated_bytes(path):
    """AllocationSize for a regular output file on the local Windows filesystem."""
    import msvcrt

    class FileStandardInfo(ctypes.Structure):
        _fields_ = [('allocation', ctypes.c_int64), ('eof', ctypes.c_int64),
                    ('links', wintypes.DWORD), ('delete_pending', ctypes.c_ubyte),
                    ('directory', ctypes.c_ubyte)]

    info = FileStandardInfo()
    query = ctypes.WinDLL('kernel32', use_last_error=True).GetFileInformationByHandleEx
    query.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    query.restype = wintypes.BOOL
    with path.open('rb') as stream:
        if not query(msvcrt.get_osfhandle(stream.fileno()), 1, ctypes.byref(info), ctypes.sizeof(info)):
            raise RuntimeError('RESOURCE_STOP: file allocation unavailable')
    return info.allocation

def require_local_fixed(*paths):
    if sys.platform != 'win32':
        raise RuntimeError('RESOURCE_STOP: Windows enforcement unavailable')
    drive_type = ctypes.WinDLL('kernel32', use_last_error=True).GetDriveTypeW
    drive_type.argtypes = [wintypes.LPCWSTR]
    drive_type.restype = wintypes.UINT
    if any(drive_type(path.absolute().anchor) != 3 for path in paths):
        raise RuntimeError('RESOURCE_STOP: local fixed drives required')
