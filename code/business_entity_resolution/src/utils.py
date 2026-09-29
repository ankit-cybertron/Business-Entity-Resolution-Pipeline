import gc
import ctypes
import os
import time
from pathlib import Path
from tqdm.auto import tqdm

from config import MEMORY_HARD_LIMIT_GB, MEMORY_WARN_LIMIT_GB

try:
    import psutil
except ImportError:
    psutil = None


def rss_gb() -> float:
    if psutil is None:
        return 0.0
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)


def memory_status(label=""):
    r = rss_gb()
    if r:
        prefix = f"[MEM] {label}: " if label else "[MEM] "
        print(f"{prefix}RSS={r:.2f} GB / hard limit={MEMORY_HARD_LIMIT_GB:.1f} GB")
    return r


def memory_guard(label="", extra_gb=0.0):
    """
    Hard stop before an allocation that would put the process above
    MEMORY_HARD_LIMIT_GB. This is deliberately conservative.
    """
    r = rss_gb()
    projected = r + max(0.0, extra_gb)
    if projected >= MEMORY_HARD_LIMIT_GB:
        raise MemoryError(
            f"Memory guard stopped '{label}'. "
            f"RSS={r:.2f} GB, requested extra≈{extra_gb:.2f} GB, "
            f"projected≈{projected:.2f} GB, hard limit={MEMORY_HARD_LIMIT_GB:.1f} GB."
        )
    if r >= MEMORY_WARN_LIMIT_GB:
        print(
            f"[MEM-WARN] {label}: RSS={r:.2f} GB is above "
            f"the {MEMORY_WARN_LIMIT_GB:.1f} GB warning level."
        )
    return True


def cleanup(label=""):
    gc.collect()
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    memory_status(f"after cleanup {label}".strip())


def progress(iterable, **kwargs):
    """Single standard progress-bar entry point for all long operations."""
    return tqdm(iterable, dynamic_ncols=True, **kwargs)


def directory_size_gb(path):
    path = Path(path)
    if not path.exists():
        return 0.0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / (1024 ** 3)
