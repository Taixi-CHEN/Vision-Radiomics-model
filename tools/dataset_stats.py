# save as tools/dataset_stats.py and run: python tools/dataset_stats.py
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path("/home/taixi/BiomedParse-finetuning/biomedparse_datasets/LUAD_WSSS")
CLASSES = ["tumor", "stroma", "normal"]

def is_binary(png_path):
    arr = np.array(Image.open(png_path))
    u = np.unique(arr)
    return set(u).issubset({0,255})

for split in ["train", "test"]:
    img_dir = ROOT / split
    mk_dir  = ROOT / f"{split}_mask"
    bases = [p.stem for p in sorted(img_dir.glob("*.png"))]
    print(f"\n=== {split.upper()} ===")
    print("images:", len(bases))

    images_with = {c: 0 for c in CLASSES}
    mask_files  = {c: 0 for c in CLASSES}
    nonbin = []

    for b in bases:
        any_for_img = {c: False for c in CLASSES}
        for c in CLASSES:
            mp = mk_dir / f"{b}_{c}.png"
            if mp.exists():
                mask_files[c] += 1
                any_for_img[c] = True
                if not is_binary(mp):
                    nonbin.append(str(mp))
        for c in CLASSES:
            if any_for_img[c]:
                images_with[c] += 1

    print("images containing class:")
    for c in CLASSES:
        print(f"  {c:7s}: {images_with[c]}")
    print("mask files per class:")
    for c in CLASSES:
        print(f"  {c:7s}: {mask_files[c]}")
    if nonbin:
        print("WARNING non-binary masks:", len(nonbin))
        print("  e.g.", nonbin[:5])



### Example output:

# === TRAIN ===
# images: 269
# images containing class:
#   tumor  : 257
#   stroma : 236
#   normal : 52
# mask files per class:
#   tumor  : 257
#   stroma : 236
#   normal : 52

# === TEST ===
# images: 47
# images containing class:
#   tumor  : 45
#   stroma : 42
#   normal : 5
# mask files per class:
#   tumor  : 45
#   stroma : 42
#   normal : 5