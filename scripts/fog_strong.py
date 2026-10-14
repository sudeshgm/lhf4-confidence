"""Stronger fog validity check. Do not train. 7B only.

Run from the repo root:
    python scripts/fog_strong.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fog import ALPHA_STRONG, apply_fog  # noqa: E402
from logutil import start_log  # noqa: E402
from smoke_vlm import PRIMARY, load_model, norm_answer, verbal_int  # noqa: E402
from exp0_controls import generate, image_messages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.json"
PRIOR = ROOT / "results" / "prior_answerable.json"
OUT_JSON = ROOT / "results" / "fog_strong.json"
IMAGE_ROOT = ROOT / "data" / "images" / "strong"
SEVERITIES = (0, 1, 2, 3, 4)
VERBAL = "How sure are you, 0-100? Reply with an integer only."


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def pearson(xs, ys) -> float:
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


def main() -> None:
    start_log(ROOT / "results" / "fog_strong.txt")
    if not PRIOR.exists():
        raise RuntimeError("results/prior_answerable.json missing. Run exp0_controls.py first.")
    prior = json.loads(PRIOR.read_text())
    drop_ids = set(prior.get("ids") or [])
    manifest = json.loads(MANIFEST.read_text())
    items = manifest["items"]
    if len(items) != 20:
        raise RuntimeError(f"expected 20 items, got {len(items)}")
    model, processor = load_model(PRIMARY, use_4bit=True)
    rows = []
    for item in items:
        clean = np.array(Image.open(ROOT / item["image"]).convert("RGB"))
        for severity in SEVERITIES:
            fogged = apply_fog(clean, severity, alphas=ALPHA_STRONG)
            folder = IMAGE_ROOT / f"sev{severity}"
            folder.mkdir(parents=True, exist_ok=True)
            image_path = folder / f"{item['id']}.png"
            Image.fromarray(fogged).save(image_path)
            pred, lp, maxprob = generate(
                model, processor, image_messages(image_path, item["question"])
            )
            pred_n = norm_answer(pred)
            vpred, _, _ = generate(
                model,
                processor,
                image_messages(image_path, VERBAL),
                max_new_tokens=8,
            )
            verbal = verbal_int(vpred)
            row = {
                "id": item["id"],
                "severity": severity,
                "image": f"data/images/strong/sev{severity}/{item['id']}.png",
                "prediction": pred_n,
                "exact_match": int(pred_n == item["gold"]),
                "logprob": lp,
                "maxprob": maxprob,
                "verbal": verbal,
                "prior_answerable": item["id"] in drop_ids,
            }
            rows.append(row)
            print(
                f"{item['id']} sev{severity} gold={item['gold']} pred={pred_n} match={row['exact_match']} lp={lp:.3f} maxp={maxprob:.3f} verbal={verbal}",
                flush=True,
            )
    scored = [r for r in rows if not r["prior_answerable"]]
    acc = {}
    conf = {}
    maxprob = {}
    verbal = {}
    for severity in SEVERITIES:
        subset = [r for r in scored if r["severity"] == severity]
        acc[severity] = mean([r["exact_match"] for r in subset])
        conf[severity] = mean([r["logprob"] for r in subset])
        maxprob[severity] = mean([r["maxprob"] for r in subset])
        verbal[severity] = mean([r["verbal"] for r in subset])
    r = pearson([conf[s] for s in SEVERITIES], [acc[s] for s in SEVERITIES])
    fell_hard = "yes" if acc[4] <= acc[0] - 0.20 else "no"
    payload = {
        "model": PRIMARY,
        "fallback": "no",
        "fog_alpha": {str(k): v for k, v in ALPHA_STRONG.items()},
        "dropped_from_means": sorted(drop_ids),
        "n_in_means": len(scored) // 5,
        "rows": rows,
        "mean_accuracy": {str(s): acc[s] for s in SEVERITIES},
        "mean_logprob": {str(s): conf[s] for s in SEVERITIES},
        "mean_maxprob": {str(s): maxprob[s] for s in SEVERITIES},
        "mean_verbal": {str(s): verbal[s] for s in SEVERITIES},
        "pearson_logprob_accuracy": r,
        "accuracy_fell_hard": fell_hard,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print("\n----- PASTE TO CPU AGENT -----")
    for s in SEVERITIES:
        print(f"SEV{s}_ACC: {acc[s]:.3f}")
    for s in SEVERITIES:
        print(f"SEV{s}_CONF: {conf[s]:.3f}")
    print(f"PEARSON: {r:.3f}")
    print(f"SEV0_MAXPROB: {maxprob[0]:.3f}")
    print(f"SEV4_MAXPROB: {maxprob[4]:.3f}")
    print(f"SEV0_VERBAL: {verbal[0]:.1f}")
    print(f"SEV4_VERBAL: {verbal[4]:.1f}")
    print(f"ACCURACY_FELL_HARD: {fell_hard}")
    print("BLOCKER: none")
    print(f"WROTE: {OUT_JSON}")
    print(f"IMAGES: {IMAGE_ROOT}")


if __name__ == "__main__":
    main()
