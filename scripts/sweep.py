"""LHF-4 five-severity sweep. Do not train. 7B only.

Run from the repo root:
    python scripts/sweep.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fog import apply_fog  # noqa: E402
from logutil import start_log  # noqa: E402
from smoke_vlm import PRIMARY, ask, load_model, norm_answer, verbal_int  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.json"
OUT = ROOT / "results" / "sweep.json"
SEVERITIES = (0, 1, 2, 3, 4)


def pearson(xs: list[float], ys: list[float]) -> float:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if len(x) != 5 or len(y) != 5:
        raise ValueError("need all five severities")
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denom == 0.0:
        return float("nan")
    return float((x * y).sum() / denom)


def mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def main() -> None:
    start_log(ROOT / "results" / "sweep.txt")
    manifest = json.loads(MANIFEST.read_text())
    items = manifest["items"]
    if len(items) != 20:
        raise RuntimeError(f"expected 20 items, got {len(items)}")
    model, processor = load_model(PRIMARY, use_4bit=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="lhf4_fog_") as tmp:
        tmp_dir = Path(tmp)
        for item in items:
            clean = np.array(Image.open(ROOT / item["image"]).convert("RGB"))
            for severity in SEVERITIES:
                fogged = apply_fog(clean, severity)
                image_path = tmp_dir / f"{item['id']}_s{severity}.png"
                Image.fromarray(fogged).save(image_path)
                pred, lp = ask(model, processor, image_path, item["question"])
                pred_n = norm_answer(pred)
                match = int(pred_n == item["gold"])
                vpred, _ = ask(
                    model,
                    processor,
                    image_path,
                    "How sure are you, 0-100? Reply with an integer only.",
                    max_new_tokens=8,
                )
                verbal = verbal_int(vpred)
                row = {
                    "id": item["id"],
                    "severity": severity,
                    "prediction": pred_n,
                    "exact_match": match,
                    "logprob": lp,
                    "verbal": verbal,
                }
                rows.append(row)
                print(
                    f"{item['id']} sev{severity} gold={item['gold']} pred={pred_n} match={match} lp={lp:.3f} verbal={verbal}",
                    flush=True,
                )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    acc = {}
    conf = {}
    verbal_means = {}
    for severity in SEVERITIES:
        subset = [r for r in rows if r["severity"] == severity]
        acc[severity] = mean([r["exact_match"] for r in subset])
        conf[severity] = mean([r["logprob"] for r in subset])
        verbal_means[severity] = mean([r["verbal"] for r in subset if r["verbal"] is not None])
    r = pearson([conf[s] for s in SEVERITIES], [acc[s] for s in SEVERITIES])
    verbal_vals = [verbal_means[s] for s in SEVERITIES]
    verbal_moved = "yes" if max(verbal_vals) - min(verbal_vals) >= 5 else "no"
    accuracy_fell = "yes" if acc[4] < acc[0] else "no"
    payload = {
        "model": PRIMARY,
        "fallback": "no",
        "n_items": 20,
        "severities": list(SEVERITIES),
        "fog_alpha": {0: 0.0, 1: 0.25, 2: 0.45, 3: 0.65, 4: 0.85},
        "rows": rows,
        "mean_accuracy": {str(s): acc[s] for s in SEVERITIES},
        "mean_logprob": {str(s): conf[s] for s in SEVERITIES},
        "mean_verbal": {str(s): verbal_means[s] for s in SEVERITIES},
        "pearson_logprob_accuracy": r,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print("\n----- PASTE TO CPU AGENT -----")
    for s in SEVERITIES:
        print(f"SEV{s}_ACC: {acc[s]:.3f}")
    for s in SEVERITIES:
        print(f"SEV{s}_CONF: {conf[s]:.3f}")
    print(f"PEARSON: {r:.3f}")
    print(f"VERBAL_MOVED: {verbal_moved}")
    print(f"ACCURACY_FELL: {accuracy_fell}")
    print("BLOCKER: none")
    print(f"WROTE: {OUT}")


if __name__ == "__main__":
    main()
