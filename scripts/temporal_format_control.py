"""Temporal format control. Do not train. 7B only.

Always feed the same number of frames as severity 0.
Severities 1-4 drop 25/50/75/100 percent, then repeat the last kept frame
until the count matches severity 0. Severity 4 is the first frame repeated.

Run from the repo root:
    export HF_HOME="$PWD/hf-cache"
    export HUGGINGFACE_HUB_CACHE="$PWD/hf-cache/hub"
    python scripts/temporal_format_control.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frame_drop import select_frames  # noqa: E402
from logutil import start_log  # noqa: E402
from smoke_vlm import PRIMARY, load_model, norm_answer, verbal_int  # noqa: E402
from temporal_smoke import ask_frames, mean  # noqa: E402
from temporal_sweep import pearson  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "temporal_manifest.json"
OUT = ROOT / "results" / "temporal_format_control.json"
SEVERITIES = (0, 1, 2, 3, 4)


def pad_to(frames: list[Path], n: int) -> list[Path]:
    if not frames:
        raise ValueError("empty frame list")
    if len(frames) >= n:
        return frames[:n]
    return frames + [frames[-1]] * (n - len(frames))


def main() -> None:
    start_log(ROOT / "results" / "temporal_format_control.txt")
    items = json.loads(MANIFEST.read_text())["items"]
    if len(items) != 8:
        raise RuntimeError(f"expected 8 clips, got {len(items)}")
    model, processor = load_model(PRIMARY, use_4bit=True)
    rows = []
    for item in items:
        frames = [ROOT / p for p in item["frames"]]
        n0 = len(frames)
        for severity in SEVERITIES:
            chosen = pad_to(select_frames(frames, severity), n0)
            if len(chosen) != n0:
                raise RuntimeError(f"{item['id']} sev{severity} has {len(chosen)} frames, expected {n0}")
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
                "n_unique": len(set(map(str, chosen))),
                "gold": item["gold"],
                "prediction": pred_n,
                "exact_match": int(pred_n == item["gold"]),
                "logprob": lp,
                "verbal": verbal,
            }
            rows.append(row)
            print(
                f"{item['id']} sev{severity} n={len(chosen)} unique={row['n_unique']} gold={item['gold']} pred={pred_n} match={row['exact_match']} lp={lp:.3f} verbal={verbal}",
                flush=True,
            )
            torch.cuda.empty_cache()
    acc, conf = [], []
    for severity in SEVERITIES:
        sev = [r for r in rows if r["severity"] == severity]
        acc.append(mean([r["exact_match"] for r in sev]))
        conf.append(mean([r["logprob"] for r in sev]))
    r = pearson(conf, acc)
    accuracy_fell = "yes" if acc[4] < acc[0] else "no"
    confidence_flat = "yes" if max(conf) - min(conf) < 0.05 else "no"
    payload = {
        "model": PRIMARY,
        "degradation": "frame_drop_padded",
        "rows": rows,
        "acc": acc,
        "conf": conf,
        "pearson": r,
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
    print(f"CONFIDENCE_FLAT: {confidence_flat}")
    print(f"ACCURACY_FELL: {accuracy_fell}")
    print("BLOCKER: none")
    print(f"WROTE: {OUT}")


if __name__ == "__main__":
    main()
