"""Temporal frame-drop smoke. Do not train. 7B only. Severities 0 and 4.

Run from the repo root:
    python scripts/temporal_smoke.py
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frame_drop import select_frames  # noqa: E402
from logutil import start_log  # noqa: E402
from smoke_vlm import PRIMARY, ask, load_model, norm_answer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "temporal_manifest.json"
OUT = ROOT / "results" / "temporal_smoke.json"


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def frames_to_video(frame_paths: list[Path], out_path: Path, fps: int = 2) -> None:
    import subprocess

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lst = out_path.with_suffix(".txt")
    lst.write_text("".join(f"file '{p.resolve()}'\nduration {1/fps}\n" for p in frame_paths))
    subprocess.check_call(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
            "-vf", "fps=2,scale=640:-2", "-pix_fmt", "yuv420p", str(out_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> None:
    start_log(ROOT / "results" / "temporal_smoke.txt")
    items = json.loads(MANIFEST.read_text())["items"]
    if len(items) != 8:
        raise RuntimeError(f"expected 8 clips, got {len(items)}")
    model, processor = load_model(PRIMARY, use_4bit=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="lhf4_temporal_") as tmp:
        tmp_dir = Path(tmp)
        for item in items:
            frames = [ROOT / p for p in item["frames"]]
            for severity in (0, 4):
                chosen = select_frames(frames, severity)
                video = tmp_dir / f"{item['id']}_s{severity}.mp4"
                frames_to_video(chosen, video)
                pred, lp = ask(model, processor, video, item["question"])
                pred_n = norm_answer(pred)
                row = {
                    "id": item["id"],
                    "severity": severity,
                    "n_frames": len(chosen),
                    "gold": item["gold"],
                    "prediction": pred_n,
                    "exact_match": int(pred_n == item["gold"]),
                    "logprob": lp,
                }
                rows.append(row)
                print(
                    f"{item['id']} sev{severity} n={len(chosen)} gold={item['gold']} pred={pred_n} match={row['exact_match']} lp={lp:.3f}",
                    flush=True,
                )
    sev0 = [r for r in rows if r["severity"] == 0]
    sev4 = [r for r in rows if r["severity"] == 4]
    payload = {
        "model": PRIMARY,
        "degradation": "frame_drop",
        "rows": rows,
        "sev0_acc": mean([r["exact_match"] for r in sev0]),
        "sev4_acc": mean([r["exact_match"] for r in sev4]),
        "sev0_conf": mean([r["logprob"] for r in sev0]),
        "sev4_conf": mean([r["logprob"] for r in sev4]),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print("\n----- PASTE TO CPU AGENT -----")
    print(f"SEV0_ACC: {payload['sev0_acc']:.3f}")
    print(f"SEV4_ACC: {payload['sev4_acc']:.3f}")
    print(f"SEV0_CONF: {payload['sev0_conf']:.3f}")
    print(f"SEV4_CONF: {payload['sev4_conf']:.3f}")
    print("BLOCKER: none")
    print(f"WROTE: {OUT}")


if __name__ == "__main__":
    main()
