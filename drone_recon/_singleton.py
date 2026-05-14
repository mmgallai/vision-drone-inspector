"""
Singleton lockfile helper for drone_recon nodes.

Prevents the "ghost-node race write" problem where a previous launch leaves
a mission_node or image_capture process alive (e.g. orphaned by Ctrl-C of
the launch wrapper but kept running by ros2 launch). When the next launch
starts a fresh instance, both nodes write to ~/recon_output simultaneously,
producing duplicate filenames and trailing-data JSON.

Usage:
    from drone_recon._singleton import acquire_singleton
    lock = acquire_singleton('image_capture')   # raises SystemExit if held

Holds an exclusive flock on /tmp/drone_recon_<name>.lock for the lifetime
of the process. The lock is released automatically on process exit.
"""

import fcntl
import os
import sys
from pathlib import Path


_LOCK_DIR = Path('/tmp')


def acquire_singleton(name: str):
    """
    Acquire an exclusive flock named after the node. If another process
    already holds it, log to stderr and exit(1) — we never want two
    instances racing on the same output directory.

    Returns the open file handle; keep it alive for the lifetime of the
    process (the kernel releases the lock on close/exit).
    """
    lockpath = _LOCK_DIR / f'drone_recon_{name}.lock'
    fh = open(lockpath, 'a+')
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        # Read the holder's PID for the error message
        fh.seek(0)
        holder = fh.read().strip() or 'unknown'
        sys.stderr.write(
            f'[drone_recon] ERROR: another {name} instance is already '
            f'running (pid={holder}, lock={lockpath}). Refusing to start '
            f'so we do not corrupt the shared output directory.\n')
        fh.close()
        sys.exit(1)

    # Truncate and write our PID so future contenders see who held it.
    fh.seek(0)
    fh.truncate()
    fh.write(f'{os.getpid()}\n')
    fh.flush()
    return fh
