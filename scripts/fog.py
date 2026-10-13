"""Deterministic grey-blend fog for the LHF-4 kill test.

severity 0: original image
severities 1-4: blend toward a fixed grey fog colour
    I = (1 - a) * I0 + a * G
    a = 0.25, 0.45, 0.65, 0.85
    G = (128, 128, 128)

No RNG. Alpha is not estimated from the image.
"""

from __future__ import annotations

import numpy as np

GREY = np.array([128.0, 128.0, 128.0], dtype=np.float32)
ALPHA = {0: 0.0, 1: 0.25, 2: 0.45, 3: 0.65, 4: 0.85}


def alpha(severity: int) -> float:
    if severity not in ALPHA:
        raise ValueError(f"severity must be in {sorted(ALPHA)}, got {severity}")
    return ALPHA[severity]


def apply_fog(image: np.ndarray, severity: int) -> np.ndarray:
    """Return uint8 RGB or RGBA. Severity 0 returns a copy."""
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("expected HxWx3 or HxWx4 image")
    a = alpha(severity)
    out = np.array(image, copy=True)
    if a == 0.0:
        return out
    rgb = out[..., :3].astype(np.float32)
    fogged = (1.0 - a) * rgb + a * GREY
    out[..., :3] = np.clip(np.rint(fogged), 0, 255).astype(np.uint8)
    return out
