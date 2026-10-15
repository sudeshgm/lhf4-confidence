"""Temporal frame-drop smoke. Do not train. 7B only. Severities 0 and 4.

Frames are sent as images, not as an mp4.

Run from the repo root:
    export HF_HOME="$PWD/hf-cache"
    export HUGGINGFACE_HUB_CACHE="$PWD/hf-cache/hub"
    python scripts/temporal_smoke.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frame_drop import select_frames  # noqa: E402
from logutil import start_log  # noqa: E402
from smoke_vlm import PRIMARY, load_model, norm_answer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "temporal_manifest.json"
OUT = ROOT / "results" / "temporal_smoke.json"


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def shrink(path: Path, long_side: int = 320) -> Image.Image:
    image = Image.open(path).convert("RGB")
    w, h = image.size
    scale = long_side / max(w, h)
    if scale < 1:
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return image


def ask_frames(model, processor, frame_paths: list[Path], question: str, max_new_tokens: int = 16):
    from qwen_vl_utils import process_vision_info

    images = [shrink(p) for p in frame_paths]
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]
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


def main() -> None:
    start_log(ROOT / "results" / "temporal_smoke.txt")
    items = json.loads(MANIFEST.read_text())["items"]
    if len(items) != 8:
        raise RuntimeError(f"expected 8 clips, got {len(items)}")
    model, processor = load_model(PRIMARY, use_4bit=True)
    rows = []
    for item in items:
        frames = [ROOT / p for p in item["frames"]]
        for severity in (0, 4):
            chosen = select_frames(frames, severity)
            pred, lp = ask_frames(model, processor, chosen, item["question"])
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
            torch.cuda.empty_cache()
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
