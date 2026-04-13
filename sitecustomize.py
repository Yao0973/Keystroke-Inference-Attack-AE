"""Repository-local Python startup customizations for artifact evaluation.

This module is imported automatically by Python's site machinery when the
repository root is on ``sys.path`` (for example when executing scripts from the
repository root).  It keeps the legacy research scripts usable in headless and
checkpoint-loading-sensitive environments without modifying each script
individually.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
ARTIFACT_CACHE = REPO_ROOT / ".artifact_cache"
MPL_CACHE = ARTIFACT_CACHE / "matplotlib"

for directory in (ARTIFACT_CACHE, MPL_CACHE):
    directory.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

try:
    import numpy as _np

    # Some shipped .pth files were serialized with a NumPy layout that refers to
    # ``numpy._core``. Older but still common NumPy releases only expose
    # ``numpy.core``.  Registering an alias keeps ``torch.load`` compatible.
    sys.modules.setdefault("numpy._core", _np.core)
except Exception:
    # Artifact scripts should still be able to fail with the real import error
    # later if NumPy is actually unavailable.
    pass
