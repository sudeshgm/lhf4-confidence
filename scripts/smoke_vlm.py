"""LHF-4 GPU smoke. Do not train.

Run from the lhf4_confidence folder:
    python scripts/smoke_vlm.py

Prints the paste block the CPU agent expects.
"""

from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.json"
PRIMARY = "Qwen/Qwen2.5-VL-7B-Instruct"
FALLBACK = "Qwen/Qwen2.5-VL-3B-Instruct"


def load_model(name: str, use_4bit: bool):
    from transformers import Qwen2_5_VLForConditionalGeneration

    kwargs = {"device_map": "auto"}
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        kwargs["torch_dtype"] = torch.float16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(name, **kwargs)
    processor = AutoProcessor.from_pretrained(name)
    model.eval()
    return model, processor


def ask(model, processor, image_path: Path, question: str, max_new_tokens: int = 16):
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": question},
            ],
        }
    ]
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
    for step, logits in enumerate(out.scores):
        if step >= len(gen_ids):
            break
        token_id = int(gen_ids[step])
        lp = torch.log_softmax(logits[0].float(), dim=-1)[token_id].item()
        logprobs.append(lp)
    mean_lp = sum(logprobs) / len(logprobs) if logprobs else float("nan")
    return pred, mean_lp


def norm_answer(text: str) -> str:
    line = text.strip().splitlines()[0].strip().lower() if text.strip() else ""
    line = line.strip(" .\"'")
    line = re.sub(r"^(answer\s*:\s*)", "", line)
    return line.split()[0] if line else ""


def verbal_int(text: str):
    m = re.search(r"\d{1,3}", text)
    return int(m.group(0)) if m else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-only", action="store_true", help="Severity 0 only. 7B only, no fallback.")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    items = manifest["items"][:8]
    model_name = PRIMARY
    fallback = "no"
    oom = "no"
    try:
        model, processor = load_model(PRIMARY, use_4bit=True)
    except Exception as exc:
        print("PRIMARY_LOAD_FAILED:", type(exc).__name__, exc)
        if args.clean_only:
            raise
        try:
            model, processor = load_model(FALLBACK, use_4bit=True)
            model_name = FALLBACK
            fallback = "yes"
        except Exception:
            traceback.print_exc()
            raise

    severities = ((0, "image"),) if args.clean_only else ((0, "image"), (4, "image_sev4"))
    rows = []
    try:
        for item in items:
            for severity, key in severities:
                image_path = ROOT / item[key]
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
                rows.append(
                    {
                        "id": item["id"],
                        "severity": severity,
                        "gold": item["gold"],
                        "pred": pred_n,
                        "match": match,
                        "logprob": lp,
                        "verbal": verbal,
                    }
                )
                print(
                    f"{item['id']} sev{severity} gold={item['gold']} pred={pred_n} match={match} lp={lp:.3f} verbal={verbal}",
                    flush=True,
                )
    except torch.cuda.OutOfMemoryError:
        oom = "yes"
        print("OOM during scoring")

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    sev0 = [r for r in rows if r["severity"] == 0]
    sev4 = [r for r in rows if r["severity"] == 4]
    print("\n----- PASTE TO CPU AGENT -----")
    print(f"MODEL: {model_name}")
    print(f"FALLBACK: {fallback}")
    print(f"N_SCORED: {len(rows)}")
    print(f"SEV0_ACCURACY: {mean([r['match'] for r in sev0]):.3f}")
    print(f"SEV4_ACCURACY: {mean([r['match'] for r in sev4]):.3f}")
    print(f"SEV0_MEAN_LOGPROB: {mean([r['logprob'] for r in sev0]):.3f}")
    print(f"SEV4_MEAN_LOGPROB: {mean([r['logprob'] for r in sev4]):.3f}")
    print(f"SEV0_MEAN_VERBAL: {mean([r['verbal'] for r in sev0]):.1f}")
    print(f"SEV4_MEAN_VERBAL: {mean([r['verbal'] for r in sev4]):.1f}")
    print(f"OOM: {oom}")
    print("ROWS:")
    for r in rows:
        print(
            f"{r['id']} | {r['severity']} | {r['gold']} | {r['pred']} | {r['match']} | {r['logprob']:.4f} | {r['verbal']}"
        )


if __name__ == "__main__":
    main()
