"""Frame-drop degradation. One degradation only.

Severity 0: original frame list.
Severities 1-4: drop 25%, 50%, 75%, 100% of the frames.
The first frame is always kept. 100% means the first frame only.
"""

from __future__ import annotations


DROP = {0: 0.0, 1: 0.25, 2: 0.50, 3: 0.75, 4: 1.0}


def select_frames(frames: list[str], severity: int) -> list[str]:
    if severity not in DROP:
        raise ValueError(f"severity must be 0-4, got {severity}")
    if not frames:
        raise ValueError("empty frame list")
    if severity == 0:
        return list(frames)
    if severity == 4 or len(frames) == 1:
        return [frames[0]]
    rest = frames[1:]
    n_drop = round(len(frames) * DROP[severity])
    n_keep_rest = max(0, len(rest) - max(0, n_drop - 0))
    # Drop from the tail of the non-first frames, evenly.
    n_keep_rest = max(0, len(frames) - n_drop - 1)
    if n_keep_rest <= 0:
        return [frames[0]]
    if n_keep_rest >= len(rest):
        return list(frames)
    step = len(rest) / n_keep_rest
    picked = [rest[min(len(rest) - 1, int(i * step))] for i in range(n_keep_rest)]
    return [frames[0]] + picked
