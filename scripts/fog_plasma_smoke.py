"""ImageNet-C fog smoke. 8 items, severities 0 and 4. Do not train.

Run from the repo root:
    python scripts/fog_plasma_smoke.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fog import IN_C_SEVERITY, apply_fog  # noqa: E402
from logutil import start_log  # noqa: E402
from smoke_vlm import PRIMARY, ask, load_model, norm_answer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.json"
PRIOR = ROOT / "results" / "prior_answerable.json"
OUT_JSON = ROOT / "results" / "fog_plasma_smoke.json"
IMAGE_ROOT = ROOT / "data" / "images" / "plasma"


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    start_log(ROOT / "results" / "fog_plasma_smoke.txt")
    drop_ids = set()
    if PRIOR.exists():
        drop_ids = set(json.loads(PRIOR.read_text()).get("ids") or [])
    items = json.loads(MANIFEST.read_text())["items"][:8]
    model, processor = load_model(PRIMARY, use_4bit=True)
    rows = []
    for item in items:
        clean = np.array(Image.open(ROOT / item["image"]).convert("RGB"))
        for severity in (0, 4):
            fogged = apply_fog(clean, severity)
            folder = IMAGE_ROOT / f"sev{severity}"
            folder.mkdir(parents=True, exist_ok=True)
            image_path = folder / f"{item['id']}.png"
            Image.fromarray(fogged).save(image_path)
            pred, lp = ask(model, processor, image_path, item["question"])
            pred_n = norm_answer(pred)
            row = {
                "id": item["id"],
                "severity": severity,
                "image": f"data/images/plasma/sev{severity}/{item['id']}.png",
                "gold": item["gold"],
                "prediction": pred_n,
                "exact_match": int(pred_n == item["gold"]),
                "logprob": lp,
                "prior_answerable": item["id"] in drop_ids,
            }
            rows.append(row)
            print(
                f"{item['id']} sev{severity} gold={item['gold']} pred={pred_n} match={row['exact_match']} lp={lp:.3f}",
                flush=True,
            )
    scored = [r for r in rows if not r["prior_answerable"]]
    sev0 = [r for r in scored if r["severity"] == 0]
    sev4 = [r for r in scored if r["severity"] == 4]
    acc0, acc4 = mean([r["exact_match"] for r in sev0]), mean([r["exact_match"] for r in sev4])
    conf0, conf4 = mean([r["logprob"] for r in sev0]), mean([r["logprob"] for r in sev4])
    fell = "yes" if acc4 < acc0 else "no"
    payload = {
        "model": PRIMARY,
        "fog": "imagenet-c plasma",
        "in_c_severity": IN_C_SEVERITY,
        "dropped_from_means": sorted(drop_ids),
        "rows": rows,
        "sev0_acc": acc0,
        "sev4_acc": acc4,
        "sev0_conf": conf0,
        "sev4_conf": conf4,
        "accuracy_fell": fell,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print("\n----- PASTE TO CPU AGENT -----")
    print("FOG_IMPL: plasma")
    print(f"SEV0_ACC: {acc0:.3f}")
    print(f"SEV4_ACC: {acc4:.3f}")
    print(f"SEV0_CONF: {conf0:.3f}")
    print(f"SEV4_CONF: {conf4:.3f}")
    print(f"ACCURACY_FELL: {fell}")
    print("STILL_READABLE: look at data/images/plasma/sev4/i01.png")
    print("BLOCKER: none")
    print(f"WROTE: {OUT_JSON}")


if __name__ == "__main__":
    main()
