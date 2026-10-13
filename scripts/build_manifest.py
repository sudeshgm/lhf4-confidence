"""Copy 20 checked Unsplash frames and write data/manifest.json."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from fog import apply_fog

ROOT = Path(__file__).resolve().parents[1]
CAND = Path("/tmp/lhf4_cands")

# Gold assigned only after visual check. Dropped frames are not listed.
ITEMS = [
    ("i01", "00_eaa3f722.jpg", "Is a steering wheel visible? Answer yes or no.", "yes"),
    ("i02", "02_7e669276.jpg", "What color is the car? Answer with one word.", "white"),
    ("i03", "03_ae79c964.jpg", "What color is the car? Answer with one word.", "black"),
    ("i04", "05_da3b142c.jpg", "What color is the car? Answer with one word.", "yellow"),
    ("i05", "06_5fe5e7ba.jpg", "What color is the car? Answer with one word.", "blue"),
    ("i06", "08_c72f17e3.jpg", "What color is the car? Answer with one word.", "red"),
    ("i07", "10_766e1452.jpg", "What color is the car? Answer with one word.", "orange"),
    ("i08", "16_bbee04a9.jpg", "What vehicle is on the road? Answer with one word.", "motorcycle"),
    ("i09", "22_84bbe3c3.jpg", "What vehicle is ahead? Answer with one word.", "bus"),
    ("i10", "24_4b100091.jpg", "Is a rider visible? Answer yes or no.", "yes"),
    ("i11", "25_00171c69.jpg", "Is a road visible ahead? Answer yes or no.", "yes"),
    ("i12", "27_0529f29e.jpg", "What vehicle is on the road? Answer with one word.", "scooter"),
    ("i13", "28_aef5c9ce.jpg", "What color is the taxi? Answer with one word.", "yellow"),
    ("i14", "31_48bbd363.jpg", "Are any vehicles visible? Answer yes or no.", "no"),
    ("i15", "32_45ae19e6.jpg", "What color is the car? Answer with one word.", "black"),
    ("i16", "33_d53bffe3.jpg", "What color is the van? Answer with one word.", "white"),
    ("i17", "36_9e4c228b.jpg", "What color is the car? Answer with one word.", "red"),
    ("i18", "38_13ff0162.jpg", "What color is the car? Answer with one word.", "blue"),
    ("i19", "40_e406faaa.jpg", "What color is the car? Answer with one word.", "green"),
    ("i20", "18_fd3cad34.jpg", "Is the car on a road? Answer yes or no.", "yes"),
]


def main() -> None:
    clean_dir = ROOT / "data" / "images" / "sev0"
    fog_dir = ROOT / "data" / "images" / "sev4"
    rows = []
    for item_id, src_name, question, gold in ITEMS:
        src = CAND / src_name
        clean = Image.open(src).convert("RGB")
        clean_path = clean_dir / f"{item_id}.png"
        clean.save(clean_path)
        arr = np.array(clean)
        ident = apply_fog(arr, 0)
        if not np.array_equal(ident, arr):
            raise RuntimeError(f"severity 0 is not identity for {item_id}")
        fogged = apply_fog(arr, 4)
        Image.fromarray(fogged).save(fog_dir / f"{item_id}.png")
        rows.append(
            {
                "id": item_id,
                "image": f"data/images/sev0/{item_id}.png",
                "image_sev4": f"data/images/sev4/{item_id}.png",
                "question": question,
                "gold": gold,
            }
        )
    manifest = {"n": len(rows), "items": rows}
    out = ROOT / "data" / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out} n={len(rows)}")


if __name__ == "__main__":
    main()
