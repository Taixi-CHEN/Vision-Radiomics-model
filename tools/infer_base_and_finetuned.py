#!/usr/bin/env python3
"""
Inference with BOTH base (HF) and finetuned checkpoints.
- Loads config: configs/biomedparse_inference.yaml
- Prompts with --classes (e.g. tumor stroma normal)
- Saves binary masks (0/255) for each model to outdir/{base,finetuned}_masks
- Optional overlays to outdir/{base,finetuned}_overlays

Usage example:
python tools/infer_base_and_finetuned.py \
  --ckpt_base hf_hub:microsoft/BiomedParse \
  --ckpt_ft /path/to/run_18/00002740/default/model_state_dict.pt \
  --input "/path/to/image_or_folder_or_glob/*.png" \
  --classes tumor stroma normal \
  --outdir /tmp/predictions --save_overlays --thresh 0.5
"""

import argparse, os, glob
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.arguments import load_opt_from_config_files
from utilities.distributed import init_distributed
from utilities.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image

# --------------------- helpers ---------------------

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

def gather_images(inp: str) -> List[Path]:
    p = Path(inp)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted([q for q in p.rglob("*") if q.suffix.lower() in IMG_EXTS])
    # treat as glob
    return sorted([Path(x) for x in glob.glob(inp, recursive=True) if Path(x).suffix.lower() in IMG_EXTS])

def safe_cls_name(s: str) -> str:
    # Keep names readable, but filesystem-safe and consistent
    return (
        s.strip()
         .replace(" ", "_")
         .replace("+", "")
         .replace("/", "-")
         .replace("\\", "-")
         .lower()
    )

def to_bool_mask(prob: np.ndarray, thresh: float) -> np.ndarray:
    # prob is float mask in [0,1] (H,W) -> bool (H,W)
    return (prob >= thresh).astype(np.uint8)

def save_mask(mask_bool: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_bool * 255).save(path)

def color_for_class(idx: int) -> Tuple[int,int,int]:
    # simple fixed palette: [red, green, blue, magenta, cyan, yellow, white, gray ...]
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 128, 255),
        (255, 0, 255), (0, 255, 255), (255, 255, 0),
        (255, 255, 255), (180, 180, 180)
    ]
    return palette[idx % len(palette)]

def make_overlay(rgb: Image.Image, masks: Dict[str, np.ndarray]) -> Image.Image:
    base = np.array(rgb).copy()
    for i, (cls, m) in enumerate(masks.items()):
        if m.dtype != np.uint8:
            m = m.astype(np.uint8)
        if m.max() == 1:  # bool-ish
            m = m * 255
        mm = m > 0
        r, g, b = color_for_class(i)
        # brighten the selected pixels, color-tinted
        base[mm, 0] = np.maximum(base[mm, 0], r)
        base[mm, 1] = (base[mm, 1] // 2 + g // 2)
        base[mm, 2] = (base[mm, 2] // 2 + b // 2)
    return Image.fromarray(base)

def load_biomed_model(ckpt: str, device: torch.device, config_path: str):
    # Build a fresh model (same arch) per checkpoint for isolation
    opt = load_opt_from_config_files([config_path])
    opt = init_distributed(opt)  # sets CUDA, device, ranks
    base = BaseModel(opt, build_model(opt)).from_pretrained(ckpt).eval().to(device)
    with torch.no_grad():
        # Cache text embeddings for speed/stability
        base.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            BIOMED_CLASSES + ["background"], is_eval=True
        )
    return base

# --------------------- main ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/biomedparse_inference.yaml")
    ap.add_argument("--ckpt_base", required=True, help="e.g. hf_hub:microsoft/BiomedParse")
    ap.add_argument("--ckpt_ft", required=True, help="path to your finetuned model_state_dict.pt")
    ap.add_argument("--input", required=True, help="image file, folder, or glob")
    ap.add_argument("--classes", nargs="+", required=True, help="e.g. tumor stroma normal")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--save_overlays", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    images = gather_images(args.input)
    if not images:
        print(f"No images found for: {args.input}")
        return

    classes = [c for c in args.classes]
    classes_safe = [safe_cls_name(c) for c in classes]

    print("Loading base model:", args.ckpt_base)
    model_base = load_biomed_model(args.ckpt_base, device, args.config)
    print("Loading finetuned model:", args.ckpt_ft)
    model_ft = load_biomed_model(args.ckpt_ft, device, args.config)

    out_base = Path(args.outdir) / "base_masks"
    out_ft   = Path(args.outdir) / "finetuned_masks"
    out_base_ov = Path(args.outdir) / "base_overlays"
    out_ft_ov   = Path(args.outdir) / "finetuned_overlays"
    out_base.mkdir(parents=True, exist_ok=True)
    out_ft.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        out_base_ov.mkdir(parents=True, exist_ok=True)
        out_ft_ov.mkdir(parents=True, exist_ok=True)

    print(f"Inferencing {len(images)} image(s) for classes: {classes}")
    with torch.no_grad():
        for img_path in images:
            try:
                pil = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[skip] failed to open {img_path}: {e}")
                continue

            # run both models
            probs_base = interactive_infer_image(model_base, pil, classes)  # list[np.ndarray HxW floats 0..1]
            probs_ft   = interactive_infer_image(model_ft,   pil, classes)

            # save per-class masks
            base_name = img_path.stem
            masks_b: Dict[str, np.ndarray] = {}
            masks_f: Dict[str, np.ndarray] = {}

            for i, cls in enumerate(classes):
                cls_safe = classes_safe[i]

                mb = to_bool_mask(np.array(probs_base[i], dtype=np.float32), args.thresh)  # (H,W) uint8 {0,1}
                mf = to_bool_mask(np.array(probs_ft[i],   dtype=np.float32), args.thresh)

                save_mask(mb, out_base / f"{base_name}_{cls_safe}.png")
                save_mask(mf, out_ft   / f"{base_name}_{cls_safe}.png")

                masks_b[cls_safe] = mb
                masks_f[cls_safe] = mf

            # optional overlays
            if args.save_overlays:
                ov_b = make_overlay(pil, masks_b)
                ov_f = make_overlay(pil, masks_f)
                ov_b.save(out_base_ov / f"{base_name}.png")
                ov_f.save(out_ft_ov   / f"{base_name}.png")

            print(f"✓ {img_path.name}  ->  {len(classes)} classes saved")

    print("\nDone. Example outputs:")
    print(" - Base masks:", out_base)
    print(" - Finetuned masks:", out_ft)
    if args.save_overlays:
        print(" - Base overlays:", out_base_ov)
        print(" - Finetuned overlays:", out_ft_ov)

if __name__ == "__main__":
    main()
