"""LHF-4 Exp 0 controls. Do not train. 7B only. Manifest unchanged.

Run from the repo root:
    python scripts/exp0_controls.py
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fog import GREY, apply_fog  # noqa: E402
from smoke_vlm import PRIMARY, load_model, norm_answer, verbal_int  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.json"
RESULTS = ROOT / "results"
VERBAL = "How sure are you, 0-100? Reply with an integer only."


def generate(model, processor, messages, max_new_tokens: int = 16):
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out.sequences[0, prompt_len:]
    pred = processor.decode(gen_ids, skip_special_tokens=True).strip()
    logprobs = []
    probs = []
    for step, logits in enumerate(out.scores):
        if step >= len(gen_ids):
            break
        dist = torch.log_softmax(logits[0].float(), dim=-1)
        token_id = int(gen_ids[step])
        lp = dist[token_id].item()
        logprobs.append(lp)
        probs.append(math.exp(lp))
    mean_lp = sum(logprobs) / len(logprobs) if logprobs else float("nan")
    maxprob = max(probs) if probs else float("nan")
    return pred, mean_lp, maxprob


def image_messages(image_path: Path, question: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": question},
            ],
        }
    ]


def text_messages(question: str):
    return [{"role": "user", "content": [{"type": "text", "text": question}]}]


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    items = manifest["items"]
    if len(items) != 20:
        raise RuntimeError(f"expected 20 items, got {len(items)}")
    model, processor = load_model(PRIMARY, use_4bit=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    prior_rows = []
    blank_rows = []
    conf_rows = []
    grey = tuple(int(x) for x in GREY.tolist())

    with tempfile.TemporaryDirectory(prefix="lhf4_exp0_") as tmp:
        tmp_dir = Path(tmp)
        for item in items:
            clean = Image.open(ROOT / item["image"]).convert("RGB")
            # 2. text-only
            pred, lp, _ = generate(model, processor, text_messages(item["question"]))
            pred_n = norm_answer(pred)
            match = int(pred_n == item["gold"])
            prior_rows.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "gold": item["gold"],
                    "prediction": pred_n,
                    "exact_match": match,
                    "logprob": lp,
                    "prior_answerable": bool(match),
                }
            )
            print(f"text {item['id']} gold={item['gold']} pred={pred_n} match={match}", flush=True)

            # 3. blank grey, same size as the clean frame
            blank = Image.new("RGB", clean.size, grey)
            blank_path = tmp_dir / f"{item['id']}_blank.png"
            blank.save(blank_path)
            pred, lp, _ = generate(model, processor, image_messages(blank_path, item["question"]))
            pred_n = norm_answer(pred)
            match = int(pred_n == item["gold"])
            blank_rows.append(
                {
                    "id": item["id"],
                    "gold": item["gold"],
                    "prediction": pred_n,
                    "exact_match": match,
                    "logprob": lp,
                    "grey": grey,
                }
            )
            print(f"blank {item['id']} gold={item['gold']} pred={pred_n} match={match} lp={lp:.3f}", flush=True)

            # 4. confidence definitions, severity 0 and 4 only
            clean_arr = np.array(clean)
            for severity in (0, 4):
                fogged = apply_fog(clean_arr, severity)
                image_path = tmp_dir / f"{item['id']}_s{severity}.png"
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
                conf_rows.append(
                    {
                        "id": item["id"],
                        "severity": severity,
                        "prediction": pred_n,
                        "exact_match": int(pred_n == item["gold"]),
                        "logprob": lp,
                        "maxprob": maxprob,
                        "verbal": verbal,
                    }
                )
                print(
                    f"conf {item['id']} sev{severity} pred={pred_n} lp={lp:.3f} maxp={maxprob:.3f} verbal={verbal}",
                    flush=True,
                )

    prior_ids = [r["id"] for r in prior_rows if r["prior_answerable"]]
    (RESULTS / "prior_answerable.json").write_text(
        json.dumps(
            {
                "model": PRIMARY,
                "n": len(prior_rows),
                "n_prior_answerable": len(prior_ids),
                "ids": prior_ids,
                "rows": prior_rows,
            },
            indent=2,
        )
        + "\n"
    )
    (RESULTS / "blank_image.json").write_text(
        json.dumps(
            {
                "model": PRIMARY,
                "grey": grey,
                "n": len(blank_rows),
                "rows": blank_rows,
            },
            indent=2,
        )
        + "\n"
    )
    (RESULTS / "confidence_defs.json").write_text(
        json.dumps(
            {
                "model": PRIMARY,
                "note": "severity 0 and 4 only. maxprob is the max token probability in the generated answer.",
                "rows": conf_rows,
            },
            indent=2,
        )
        + "\n"
    )

    text_acc = mean([r["exact_match"] for r in prior_rows])
    blank_acc = mean([r["exact_match"] for r in blank_rows])
    blank_conf = mean([r["logprob"] for r in blank_rows])
    sev0 = [r for r in conf_rows if r["severity"] == 0]
    sev4 = [r for r in conf_rows if r["severity"] == 4]
    excluded = len(prior_ids) / len(prior_rows)

    print("\n----- PASTE TO CPU AGENT -----")
    print(f"TEXT_ONLY_ACC: {text_acc:.3f}")
    print(f"N_PRIOR_ANSWERABLE: {len(prior_ids)}")
    print(f"BLANK_ACC: {blank_acc:.3f}")
    print(f"BLANK_CONF: {blank_conf:.3f}")
    print(f"SEV0_MAXPROB: {mean([r['maxprob'] for r in sev0]):.3f}")
    print(f"SEV4_MAXPROB: {mean([r['maxprob'] for r in sev4]):.3f}")
    print(f"SEV0_VERBAL: {mean([r['verbal'] for r in sev0]):.1f}")
    print(f"SEV4_VERBAL: {mean([r['verbal'] for r in sev4]):.1f}")
    print(f"EXCLUDED_FRACTION: {excluded:.3f}")
    print("BLOCKER: none")


if __name__ == "__main__":
    main()
