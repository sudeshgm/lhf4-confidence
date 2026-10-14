"""ImageNet-C fog. Grey blend is retired.

Severity 0: original image.
Severities 1-4 map to ImageNet-C fog severities 1, 2, 3, 5.
Same transform: diamond-square plasma, low-frequency, severity-scaled contrast crush.
No blur. No grey blend. Seeded so a severity is fixed.
"""

from __future__ import annotations

import numpy as np

# Retired grey blend. Not used.
GREY = np.array([128.0, 128.0, 128.0], dtype=np.float32)
ALPHA = {0: 0.0, 1: 0.25, 2: 0.45, 3: 0.65, 4: 0.85}
ALPHA_STRONG = {0: 0.0, 1: 0.50, 2: 0.70, 3: 0.85, 4: 0.95}

# Our severity -> ImageNet-C severity.
IN_C_SEVERITY = {1: 1, 2: 2, 3: 3, 4: 5}
# (fog amount, wibbledecay). From Hendrycks ImageNet-C.
IN_C_PARAMS = {
    1: (1.5, 2.0),
    2: (2.0, 2.0),
    3: (2.5, 1.7),
    4: (2.5, 1.5),
    5: (3.0, 1.4),
}
SEED = 0


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def plasma_fractal(mapsize: int = 256, wibbledecay: float = 3.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Diamond-square heightmap in 0-1. mapsize must be a power of two."""
    if mapsize & (mapsize - 1) != 0:
        raise ValueError("mapsize must be a power of two")
    rng = rng or np.random.default_rng(SEED)
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0.0
    stepsize = mapsize
    wibble = 100.0

    def wibbledmean(array: np.ndarray) -> np.ndarray:
        return array / 4.0 + wibble * rng.uniform(-wibble, wibble, array.shape)

    def fillsquares() -> None:
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum = squareaccum + np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds() -> None:
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ldrsum + lulsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(tdrsum + tulsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    peak = maparray.max()
    if peak == 0:
        return np.zeros_like(maparray)
    return maparray / peak


def apply_fog(image: np.ndarray, severity: int) -> np.ndarray:
    """ImageNet-C fog. Severity 0 is a copy. 1-4 map to ImageNet-C 1, 2, 3, 5."""
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("expected HxWx3 or HxWx4 image")
    out = np.array(image, copy=True)
    if severity == 0:
        return out
    if severity not in IN_C_SEVERITY:
        raise ValueError(f"severity must be 0-4, got {severity}")
    amount, decay = IN_C_PARAMS[IN_C_SEVERITY[severity]]
    h, w = out.shape[:2]
    rng = np.random.default_rng(SEED + severity)
    plasma = plasma_fractal(_next_pow2(max(h, w, 2)), wibbledecay=decay, rng=rng)
    plasma = plasma[:h, :w]
    rgb = out[..., :3].astype(np.float64) / 255.0
    max_val = float(rgb.max())
    fogged = rgb + amount * plasma[..., None]
    fogged = np.clip(fogged * max_val / (max_val + amount), 0, 1) * 255.0
    out[..., :3] = np.rint(fogged).astype(np.uint8)
    return out
