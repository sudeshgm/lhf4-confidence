"""Temporal frame-drop sweep. Do not train. 7B only. Severities 0-4.

Run from the repo root:
    export HF_HOME="$PWD/hf-cache"
    export HUGGINGFACE_HUB_CACHE="$PWD/hf-cache/hub"
    python scripts/temporal_sweep.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logutil import start_log  # noqa: E402
from smoke_vlm import PRIMARY, load_model, norm_answer, verbal_int  # noqa: E402
from temporal_smoke import ask_frames, mean  # noqa: E402
from frame_drop import select_frames  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "temporal_manifest.json"
OUT = ROOT / "results" / "temporal_sweep.json"
SEVERITIES = (0, 1, 2, 3, 4)


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def main() -> None:
    start_log(ROOT / "results" / "temporal_sweep.txt")
    items = json.loads(MANIFEST.read_text())["items"]
    if len(items) != 8:
        raise RuntimeError(f"expected 8 clips, got {len(items)}")
    model, processor = load_model(PRIMARY, use_4bit=True)
    rows = []
    for item in items:
        frames = [ROOT / p for p in item["frames"]]
        for severity in SEVERITIES:
            chosen = select_frames(frames, severity)
            pred, lp = ask_frames(model, processor, chosen, item["question"])
            pred_n = norm_answer(pred)
            vpred, _ = ask_frames(
                model,
                processor,
                chosen,
                "How sure are you, 0-100? Reply with an integer only.",
                max_new_tokens=8,
            )
            verbal = verbal_int(vpred)
            row = {
                "id": item["id"],
                "severity": severity,
                "n_frames": len(chosen),
                "gold": item["gold"],
                "prediction": pred_n,
                "exact_match": int(pred_n == item["gold"]),
                "logprob": lp,
                "verbal": verbal,
            }
            rows.append(row)
            print(
                f"{item['id']} sev{severity} n={len(chosen)} gold={item['gold']} pred={pred_n} match={row['exact_match']} lp={lp:.3f} verbal={verbal}",
                flush=True,
            )
            torch.cuda.empty_cache()
    acc, conf, verbal = [], [], []
    for severity in SEVERITIES:
        sev = [r for r in rows if r["severity"] == severity]
        acc.append(mean([r["exact_match"] for r in sev]))
        conf.append(mean([r["logprob"] for r in sev]))
        verbal.append(mean([r["verbal"] for r in sev]))
    r = pearson(conf, acc)
    verbal_moved = "yes" if abs(verbal[4] - verbal[0]) >= 5 else "no"
    accuracy_fell = "yes" if acc[4] < acc[0] else "no"
    confidence_flat = "yes" if max(conf) - min(conf) < 0.05 else "no"
    payload = {
        "model": PRIMARY,
        "degradation": "frame_drop",
        "rows": rows,
        "acc": acc,
        "conf": conf,
        "verbal": verbal,
        "pearson": r,
        "verbal_moved": verbal_moved,
        "accuracy_fell": accuracy_fell,
        "confidence_flat": confidence_flat,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print("\n----- PASTE TO CPU AGENT -----")
    for i in SEVERITIES:
        print(f"SEV{i}_ACC: {acc[i]:.3f}")
    for i in SEVERITIES:
        print(f"SEV{i}_CONF: {conf[i]:.3f}")
    print(f"PEARSON: {r:.3f}")
    print(f"VERBAL_MOVED: {verbal_moved}")
    print(f"ACCURACY_FELL: {accuracy_fell}")
    print(f"CONFIDENCE_FLAT: {confidence_flat}")
    print("BLOCKER: none")
    print(f"WROTE: {OUT}")


if __name__ == "__main__":
    main()
