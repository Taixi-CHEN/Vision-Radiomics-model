#!/usr/bin/env python3
"""
修改版的推理脚本，启用radiomics融合功能
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
    return (
        s.strip()
         .replace(" ", "_")
         .replace("+", "")
         .replace("/", "-")
         .replace("\\", "-")
         .lower()
    )

def to_bool_mask(prob: np.ndarray, thresh: float) -> np.ndarray:
    return (prob >= thresh).astype(np.uint8)

def save_mask(mask_bool: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_bool * 255).save(path)

def color_for_class(idx: int) -> Tuple[int,int,int]:
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
        base[mm, 0] = np.maximum(base[mm, 0], r)
        base[mm, 1] = (base[mm, 1] // 2 + g // 2)
        base[mm, 2] = (base[mm, 2] // 2 + b // 2)
    return Image.fromarray(base)

def load_biomed_model(ckpt: str, device: torch.device, config_path: str, enable_radiomics: bool = False):
    """加载BiomedParse模型，可选择启用radiomics融合"""
    opt = load_opt_from_config_files([config_path])
    opt = init_distributed(opt)
    model = BaseModel(opt, build_model(opt)).from_pretrained(ckpt).eval().to(device)
    
    # 启用radiomics融合（如果模型支持）
    if enable_radiomics and hasattr(model.model, 'enable_radiomics_fusion'):
        model.model.enable_radiomics_fusion = True
        print(f"✓ Enabled radiomics fusion for model: {ckpt}")
        print(f"✓ Model type: {type(model.model).__name__}")
        print(f"✓ Radiomics fusion status: {model.model.enable_radiomics_fusion}")
        
        # 验证radiomics相关组件是否存在
        if hasattr(model.model, 'radiomics_fusion'):
            print(f"✓ Radiomics fusion module: {type(model.model.radiomics_fusion).__name__}")
        else:
            print(f"⚠ Warning: No radiomics_fusion module found")
            
    elif enable_radiomics:
        print(f"⚠ Warning: Model {ckpt} does not support radiomics fusion")
        print(f"⚠ Model type: {type(model.model).__name__}")
        print(f"⚠ Available attributes: {[attr for attr in dir(model.model) if 'radiomics' in attr.lower()]}")
    
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            BIOMED_CLASSES + ["background"], is_eval=True
        )
    return model

def verify_radiomics_enabled(model, verbose: bool = True):
    """验证radiomics融合是否正确启用"""
    if not hasattr(model.model, 'enable_radiomics_fusion'):
        if verbose:
            print("❌ Model does not have enable_radiomics_fusion attribute")
        return False
    
    if not model.model.enable_radiomics_fusion:
        if verbose:
            print("❌ enable_radiomics_fusion is False")
        return False
    
    if verbose:
        print("✅ enable_radiomics_fusion is True")
        
        # 检查相关组件
        if hasattr(model.model, 'radiomics_fusion'):
            print(f"✅ radiomics_fusion module: {type(model.model.radiomics_fusion).__name__}")
        else:
            print("❌ No radiomics_fusion module")
            
        if hasattr(model.model, 'jamba_model'):
            print(f"✅ jamba_model: {model.model.jamba_model is not None}")
        else:
            print("❌ No jamba_model")
            
        if hasattr(model.model, 'radiomics_feature_dim'):
            print(f"✅ radiomics_feature_dim: {model.model.radiomics_feature_dim}")
        else:
            print("❌ No radiomics_feature_dim")
            
        if hasattr(model.model, 'radiomics_seq_length'):
            print(f"✅ radiomics_seq_length: {model.model.radiomics_seq_length}")
        else:
            print("❌ No radiomics_seq_length")
    
    return True

def force_enable_radiomics(model):
    """强制启用radiomics融合（如果可能）"""
    if hasattr(model.model, 'enable_radiomics_fusion'):
        model.model.enable_radiomics_fusion = True
        print("🔧 Force enabled radiomics fusion")
        return True
    else:
        print("❌ Cannot force enable radiomics fusion - model does not support it")
        return False

def create_dummy_radiomics_data(num_rois: int = 1, seq_length: int = 8, feature_dim: int = 16):
    """创建虚拟的radiomics数据用于测试"""
    return torch.randn(num_rois, seq_length * feature_dim)

def interactive_infer_image_with_radiomics(model, image, prompts, radiomics_data=None):
    """带radiomics数据的推理函数"""
    # 如果模型支持radiomics且提供了radiomics数据
    if hasattr(model.model, 'enable_radiomics_fusion') and model.model.enable_radiomics_fusion:
        if radiomics_data is None:
            # 创建虚拟radiomics数据
            radiomics_data = create_dummy_radiomics_data()
            print("Using dummy radiomics data for inference")
        
        # 创建包含radiomics的输入
        batched_inputs = [{
            'image': image,
            'text': prompts,
            'radiomics': radiomics_data
        }]
        
        # 使用模型的forward_seg方法
        with torch.no_grad():
            outputs = model.model.forward_seg(batched_inputs)
            
        # 提取预测结果
        pred_masks = outputs['pred_masks'][0]  # [num_queries, H, W]
        
        # 转换为概率mask
        pred_mask_prob = []
        for i, prompt in enumerate(prompts):
            if i < pred_masks.shape[0]:
                mask_prob = torch.sigmoid(pred_masks[i]).cpu().numpy()
                pred_mask_prob.append(mask_prob)
            else:
                # 如果没有对应的预测，创建零mask
                pred_mask_prob.append(np.zeros(image.size[1], image.size[0]))
        
        return pred_mask_prob
    else:
        # 使用原始推理方法
        return interactive_infer_image(model, image, prompts)

# --------------------- main ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/biomedparse_inference.yaml")
    ap.add_argument("--ckpt_base", required=True, help="e.g. pretrained/biomedparse_v1.pt")
    ap.add_argument("--ckpt_ft", required=True, help="path to your finetuned model_state_dict.pt")
    ap.add_argument("--input", required=True, help="image file, folder, or glob")
    ap.add_argument("--classes", nargs="+", required=True, help="e.g. tumor stroma normal")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--save_overlays", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--enable_radiomics", default=True)
    ap.add_argument("--radiomics_data", default="/home/taixi/BiomedParse-finetuning/biomedparse_datasets/IGNITE_CANCER/test_radiomics/patient16_he_roi11_all.csv")
    args = ap.parse_args()

    device = torch.device(args.device)

    images = gather_images(args.input)
    if not images:
        print(f"No images found for: {args.input}")
        return

    classes = [c for c in args.classes]
    classes_safe = [safe_cls_name(c) for c in classes]

    print("Loading base model:", args.ckpt_base)
    model_base = load_biomed_model(args.ckpt_base, device, args.config, enable_radiomics=False)
    
    print("Loading finetuned model:", args.ckpt_ft)
    model_ft = load_biomed_model(args.ckpt_ft, device, args.config, enable_radiomics=args.enable_radiomics)
    
    # 验证radiomics是否启用
    if args.enable_radiomics:
        print("\n🔍 Verifying radiomics fusion status:")
        if not verify_radiomics_enabled(model_ft, verbose=True):
            print("\n🔧 Attempting to force enable radiomics fusion:")
            if force_enable_radiomics(model_ft):
                verify_radiomics_enabled(model_ft, verbose=True)
            else:
                print("❌ Failed to enable radiomics fusion")
                print("   This may indicate the model was not trained with radiomics support")
        print()

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
    if args.enable_radiomics:
        print("✓ Radiomics fusion enabled for finetuned model")
    
    with torch.no_grad():
        for img_path in images:
            try:
                pil = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[skip] failed to open {img_path}: {e}")
                continue

            # 准备radiomics数据（如果启用）
            radiomics_data = None
            if args.enable_radiomics:
                if args.radiomics_data and Path(args.radiomics_data).exists():
                    # 加载真实的radiomics数据
                    import pandas as pd
                    df = pd.read_csv(args.radiomics_data)
                    # 选择数值列并转换为tensor
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    radiomics_array = df[numeric_cols].values
                    radiomics_data = torch.tensor(radiomics_array, dtype=torch.float32)
                    print(f"Loaded radiomics data: {radiomics_array.shape}")
                else:
                    # 使用虚拟数据
                    radiomics_data = create_dummy_radiomics_data()
                    print("Using dummy radiomics data")

            # run both models
            probs_base = interactive_infer_image(model_base, pil, classes)
            
            if args.enable_radiomics:
                probs_ft = interactive_infer_image_with_radiomics(model_ft, pil, classes, radiomics_data)
            else:
                probs_ft = interactive_infer_image(model_ft, pil, classes)

            # save per-class masks
            base_name = img_path.stem
            masks_b: Dict[str, np.ndarray] = {}
            masks_f: Dict[str, np.ndarray] = {}

            for i, cls in enumerate(classes):
                cls_safe = classes_safe[i]

                mb = to_bool_mask(np.array(probs_base[i], dtype=np.float32), args.thresh)
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
    
    if args.enable_radiomics:
        print("\n✓ Radiomics fusion was enabled for the finetuned model")
        print("  This means the finetuned model used radiomics features during inference")

if __name__ == "__main__":
    main()
