"""Render separate result molecules without sharing Matplotlib state."""

import __main__
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from itertools import repeat
from multiprocessing import current_process, get_context
from pathlib import Path


def _init_plot_worker():
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")


def _available_memory():
    if os.name == "nt":
        import ctypes
        from ctypes.wintypes import DWORD

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", DWORD),
                ("load", DWORD),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_pagefile", ctypes.c_ulonglong),
                ("available_pagefile", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.available_physical
    elif hasattr(os, "sysconf"):
        try:
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError):
            pass
    return None


def _worker_count(row_count):
    cpus = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    workers = min(8, cpus, max(4, cpus // 2), max(1, row_count // 2))
    memory = _available_memory()
    if memory is not None and memory >= 0:
        mib = 1024**2
        workers = min(workers, max(1, (memory - 768 * mib) // (256 * mib)))
    return workers


def render_molecules(draw, results, output_dir):
    rows = list(results)
    workers = _worker_count(len(rows))
    main_file = getattr(__main__, "__file__", "")
    if (
        workers > 1
        and len(rows) >= 12
        and len({row[0] for row in rows}) == len(rows)
        and main_file
        and Path(main_file).is_file()
        and not getattr(sys, "frozen", False)
        and not current_process().daemon
    ):
        groups = [rows[i::workers] for i in range(workers)]
        try:
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=get_context("spawn"), initializer=_init_plot_worker
            ) as pool:
                for _ in pool.map(draw, groups, repeat(output_dir)):
                    pass
            return
        except (BrokenProcessPool, OSError):
            # Complete the original output set if worker startup is unavailable.
            pass
    draw(rows, output_dir)
