"""
Multiprocessing and threading configurations
"""

import logging
import multiprocessing as mp
import os
import sys


logger = logging.getLogger(__name__)


# Force "spawn" mode on Windows and in Jupyter Notebook environments to avoid forking issues
if sys.platform.startswith("win") or "ipykernel" in sys.modules:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        # mp.set_start_method raises RuntimeError if the start method was already set
        logger.info(f"Multiprocessing start method already set: {e}")
    except Exception as e:
        logger.error(f"Error while setting the multiprocessing start method: {e}")


# Limit BLAS/OpenMP threads in worker processes to avoid oversubscription and crashes
_DEFAULT_WORKER_MAX_THREADS = 1


def init_workers(max_threads: int = _DEFAULT_WORKER_MAX_THREADS):
    max_threads = max(1, int(max_threads))

    os.environ.setdefault("OMP_NUM_THREADS", str(max_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(max_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(max_threads))
    try:
        # Add optional threadpool control (best-effort)
        from threadpoolctl import threadpool_limits  # type: ignore
        if threadpool_limits is not None:
            threadpool_limits(max_threads=max_threads)
    except Exception:
        threadpool_limits = None
