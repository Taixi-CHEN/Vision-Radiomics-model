#!/usr/bin/env python3
import os, sys, argparse, json, math, glob
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
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

# ---------- helpers: misc ----------
def imread_gray_bool(p):
    """Load mask as {0,1} np.uint8."""
    arr = np.array(Image.open(p).convert('L'))
    if arr.max() > 1:
        arr = (arr > 127).astype(np.uint8)
    return arr

def to_bool(a): return (np.asarray(a) > 0).astype(np.uint8)

def safe_div(n, d): return (n / d) if d != 0 else 0.0


def _rankdata_avg_ties(x: np.ndarray) -> np.ndarray:
    """
    Return average ranks (1..N) with ties averaged, NumPy-only.
    """
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)  # 1-based provisional ranks

    # average ranks for ties
    x_sorted = x[order]
    # indices where value changes
    diffs = np.diff(x_sorted)
    boundaries = np.flatnonzero(diffs != 0) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(x_sorted)]))
    for s, e in zip(starts, ends):
        if e - s > 1:
            mean_rank = ranks[order[s:e]].mean()
            ranks[order[s:e]] = mean_rank
    return ranks

def roc_auc_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute ROC-AUC using Mann–Whitney relation with average ranks.
    Returns NaN if y_true lacks both classes.
    """
    y_true = y_true.astype(np.uint8).ravel()
    y_score = y_score.astype(np.float64).ravel()

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')

    ranks = _rankdata_avg_ties(y_score)  # 1..N with ties averaged
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def metrics_basic(pred, gt):
    """pred, gt are {0,1} np.uint8"""
    pred = pred.astype(np.uint8)
    gt   = gt.astype(np.uint8)
    # Eval region = union to avoid background dominance
    region = (pred | gt).astype(bool)
    if region.sum() == 0:
        # Nothing to evaluate (both empty)
        return dict(
            acc=1.0, prec=1.0, rec=1.0, dice=1.0, iou=1.0,
            tp=0, fp=0, fn=0, tn=0, area_pred=0, area_gt=0, eval_pix=0
        )
    P = pred[region]; G = gt[region]
    tp = int(((P==1)&(G==1)).sum())
    tn = int(((P==0)&(G==0)).sum())
    fp = int(((P==1)&(G==0)).sum())
    fn = int(((P==0)&(G==1)).sum())
    acc  = safe_div(tp+tn, len(P))
    prec = safe_div(tp, tp+fp)
    rec  = safe_div(tp, tp+fn)
    dice = safe_div(2*tp, 2*tp+fp+fn)
    iou  = safe_div(tp, tp+fp+fn)
    return dict(
        acc=float(acc), prec=float(prec), rec=float(rec),
        dice=float(dice), iou=float(iou),
        tp=tp, fp=fp, fn=fn, tn=tn,
        area_pred=int(pred.sum()), area_gt=int(gt.sum()),
        eval_pix=int(region.sum())
    )

# ---------- helpers: surface distances (ASSD, HD95) ----------
def _try_import_distance():
    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt
        return 'scipy'
    except Exception:
        try:
            from skimage.morphology import binary_erosion
            from scipy.ndimage import distance_transform_edt
            return 'mixed'
        except Exception:
            return 'none'

_DIST_IMPL = _try_import_distance()
if _DIST_IMPL in ('scipy','mixed'):
    from scipy.ndimage import distance_transform_edt
    try:
        from scipy.ndimage import binary_erosion
    except Exception:
        from skimage.morphology import binary_erosion

def _boundary(a):
    a = a.astype(bool)
    if a.sum() == 0:
        return np.zeros_like(a, dtype=bool)
    er = binary_erosion(a)
    return a & (~er)

def surface_distances(pred, gt):
    """Return per-boundary distances from pred->gt and gt->pred (pixels)."""
    if _DIST_IMPL == 'none':
        # Fallback: empty -> no distances, mark NaN
        return np.array([]), np.array([])
    Pb = _boundary(pred); Gb = _boundary(gt)
    # If one side has no boundary, return empty to signal NaN
    if Pb.sum() == 0 or Gb.sum() == 0:
        return np.array([]), np.array([])
    # distance to nearest boundary pixel: use EDT on complement of boundary
    # EDT returns distance to nearest zero; so make boundary=0, non-boundary=1
    dt_G = distance_transform_edt(~Gb)
    dt_P = distance_transform_edt(~Pb)
    d_P_to_G = dt_G[Pb]  # distances of P boundary pixels to G boundary
    d_G_to_P = dt_P[Gb]
    return d_P_to_G.astype(np.float64), d_G_to_P.astype(np.float64)

def assd_hd95(pred, gt):
    """ASSD and Hausdorff95 (symmetric). Returns (assd, hd95) or (nan, nan) if undefined."""
    d1, d2 = surface_distances(pred, gt)
    if d1.size == 0 or d2.size == 0:
        return (float('nan'), float('nan'))
    all_d = np.concatenate([d1, d2])
    assd = float(all_d.mean())
    hd95 = float(np.percentile(all_d, 95))
    return assd, hd95

# ---------- model loading & inference ----------
def load_model(ckpt, device='cuda'):
    opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])
    opt = init_distributed(opt)
    base = BaseModel(opt, build_model(opt))
    base = base.from_pretrained(ckpt).eval().to(device)
    with torch.no_grad():
        base.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            BIOMED_CLASSES + ["background"], is_eval=True
        )
    return base

def auc_on_union_region(prob: np.ndarray, gt: np.ndarray) -> float:
    """
    Evaluate ROC-AUC on union(pred, gt) region to mirror other metrics' region.
    Here we don't know pred (thresholded) yet, so use prob>0 for union proxy.
    """
    prob = prob.astype(np.float32)
    gt = (gt > 0).astype(np.uint8)
    region = ((prob > 0) | (gt > 0))
    if region.sum() == 0:
        return float('nan')
    return roc_auc_from_scores(gt[region], prob[region])


def predict_classes(model, image_pil, classes, thresh=0.5):
    """Returns dict {cls: (prob_float32 HxW, bin_uint8 HxW)} in image size."""
    probs = interactive_infer_image(model, image_pil, classes)  # list of HxW float
    out = {}
    for cls, pr in zip(classes, probs):
        pr = np.asarray(pr, dtype=np.float32)
        bm = (pr >= thresh).astype(np.uint8)
        out[cls] = (pr, bm)
    return out

# ---------- main eval ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True,
                    help=".../biomedparse_datasets/LUAD_WSSS")
    ap.add_argument("--ckpt_base", default="hf_hub:microsoft/BiomedParse")
    ap.add_argument("--ckpt_finetuned", required=True)
    ap.add_argument("--classes", nargs="+", default=["tumor","stroma","normal"])
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--save_preds", action="store_true")
    ap.add_argument("--save_overlays", action="store_true")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    test_dir = root / "test"
    gt_dir   = root / "test_mask"
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Collect test images
    images = sorted(test_dir.glob("*.png"))
    if not images:
        print(f"No test images in {test_dir}")
        sys.exit(1)

    # Load models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading base model: {args.ckpt_base}")
    model_base = load_model(args.ckpt_base, device)
    print(f"Loading finetuned model: {args.ckpt_finetuned}")
    model_ft   = load_model(args.ckpt_finetuned, device)

    rows = []  # per-image metrics
    micro = {} # pooled counts by (model, cls)
    macro = {} # sum of metric lists for mean later

    def push_agg(dct, key, val):
        dct.setdefault(key, []).append(val)

    for imgp in images:
        base_name = imgp.stem
        im = Image.open(imgp).convert("RGB")

        # Run inference once per model
        preds_base = predict_classes(model_base, im, args.classes, thresh=args.thresh)
        preds_ft   = predict_classes(model_ft,   im, args.classes, thresh=args.thresh)

        for cls in args.classes:
            gtp = gt_dir / f"{base_name}_{cls}.png"
            if not gtp.exists():
                # no GT for this class on this image; skip
                continue
            gt = imread_gray_bool(gtp)

            for model_tag, pred_pack in [("base", preds_base[cls]), ("finetuned", preds_ft[cls])]:
                pr_prob, pr_bin = pred_pack
                mb = metrics_basic(pr_bin, gt)
                assd, hd95 = assd_hd95(pr_bin, gt)
                auc = auc_on_union_region(pr_prob, gt)  
                row = {
                    "image": base_name, "class": cls, "model": model_tag,
                    "acc": mb["acc"], "precision": mb["prec"], "recall": mb["rec"],
                    "dice": mb["dice"], "iou": mb["iou"],
                    "ASSD": assd, "HD95": hd95,
                    "tp": mb["tp"], "fp": mb["fp"], "fn": mb["fn"], "tn": mb["tn"],
                    "pred_area": mb["area_pred"], "gt_area": mb["area_gt"], "eval_pixels": mb["eval_pix"],
                    "roc_auc": auc  
                }
                rows.append(row)

                # micro pool counts (unchanged)
                mkey = (model_tag, cls)
                m = micro.setdefault(mkey, dict(tp=0,fp=0,fn=0,tn=0,eval=0, auc_scores=[], auc_truth=[]))
                m["tp"]  += mb["tp"]; m["fp"] += mb["fp"]; m["fn"] += mb["fn"]; m["tn"] += mb["tn"]; m["eval"] += mb["eval_pix"]

                # micro AUC pooling: store raw scores and labels over the union region
                reg = ((pr_prob > 0) | (gt > 0))
                if reg.sum() > 0:
                    m["auc_scores"].append(pr_prob[reg].astype(np.float32).ravel())
                    m["auc_truth"].append((gt[reg] > 0).astype(np.uint8).ravel())

# macro lists (existing) + add AUC
               
                # macro lists
                push_agg(macro, (model_tag, cls, "acc"),   mb["acc"])
                push_agg(macro, (model_tag, cls, "prec"),  mb["prec"])
                push_agg(macro, (model_tag, cls, "rec"),   mb["rec"])
                push_agg(macro, (model_tag, cls, "dice"),  mb["dice"])
                push_agg(macro, (model_tag, cls, "iou"),   mb["iou"])
                push_agg(macro, (model_tag, cls, "ASSD"),  assd)
                push_agg(macro, (model_tag, cls, "HD95"),  hd95)
                push_agg(macro, (model_tag, cls, "AUC"), auc)

                # save predictions / overlays
                if args.save_preds:
                    pdir = outdir / "pred_masks" / model_tag
                    pdir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray((pr_bin*255).astype(np.uint8)).save(pdir / f"{base_name}_{cls}.png")

                if args.save_overlays:
                    odir = outdir / "overlays" / model_tag
                    odir.mkdir(parents=True, exist_ok=True)
                    # green = GT, red = Pred
                    base_np = np.array(im).copy()
                    gt_rgb = (to_bool(gt)>0)
                    pr_rgb = (pr_bin>0)
                    # draw GT in green
                    base_np[gt_rgb, 1] = 255
                    base_np[gt_rgb, 0] = base_np[gt_rgb, 0]//2
                    base_np[gt_rgb, 2] = base_np[gt_rgb, 2]//2
                    # draw Pred in red (overrides mix)
                    base_np[pr_rgb, 0] = 255
                    base_np[pr_rgb, 1] = base_np[pr_rgb, 1]//2
                    base_np[pr_rgb, 2] = base_np[pr_rgb, 2]//2
                    Image.fromarray(base_np).save(odir / f"{base_name}_{cls}.png")

    # Write per-image CSV
    import csv
    per_csv = outdir / "metrics_per_image.csv"
    with open(per_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["image","class","model","acc","precision","recall","dice","iou","ASSD","HD95","tp","fp","fn","tn","pred_area","gt_area","eval_pixels", "roc_auc"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Compute macro/micro summaries per class and overall
    def nanmean(x):
        a = np.array(x, dtype=float)
        return float(np.nanmean(a)) if a.size else float('nan')

    summary_rows = []
    models = ["base","finetuned"]
    classes = args.classes + ["ALL"]

    for model_tag in models:
        for cls in classes:
            # macro (mean of per-image metrics)
            if cls != "ALL":
                mac = dict(
                    acc=nanmean(macro.get((model_tag, cls, "acc"), [])),
                    precision=nanmean(macro.get((model_tag, cls, "prec"), [])),
                    recall=nanmean(macro.get((model_tag, cls, "rec"), [])),
                    dice=nanmean(macro.get((model_tag, cls, "dice"), [])),
                    iou=nanmean(macro.get((model_tag, cls, "iou"), [])),
                    ASSD=nanmean(macro.get((model_tag, cls, "ASSD"), [])),
                    HD95=nanmean(macro.get((model_tag, cls, "HD95"), [])),
                    AUC=nanmean(macro.get((model_tag, cls, "AUC"), [])),
                )
            else:
                # average of class macros
                vals = {k:[] for k in ["acc","precision","recall","dice","iou","ASSD","HD95", "AUC"]}
                for c in args.classes:
                    for k in vals.keys():
                        vals[k].append(nanmean(macro.get((model_tag, c, k if k in ["ASSD","HD95"] else {"acc":"acc","precision":"prec","recall":"rec","dice":"dice","iou":"iou"}[k]), [])))
                mac = {k: nanmean(v) for k,v in vals.items()}

            # micro (pooled counts)
            if cls != "ALL":
                m = micro.get((model_tag, cls), dict(tp=0,fp=0,fn=0,tn=0,eval=0,auc_scores=[], auc_truth=[]))
                tp,fp,fn,tn = m["tp"],m["fp"],m["fn"],m["tn"]
                if m.get("auc_scores"):
                    ys = np.concatenate(m["auc_scores"])
                    yt = np.concatenate(m["auc_truth"])
                    micro_auc_val = roc_auc_from_scores(yt, ys)
                else:
                    micro_auc_val = float('nan')
            else:
                tp=fp=fn=tn=0
                auc_scores_all = []
                auc_truth_all = []
                for c in args.classes:
                    m = micro.get((model_tag, c), dict(tp=0,fp=0,fn=0,tn=0,eval=0, auc_scores=[], auc_truth=[]))
                    tp+=m["tp"]; fp+=m["fp"]; fn+=m["fn"]; tn+=m["tn"]
                    if m.get("auc_scores"):
                        auc_scores_all.extend(m["auc_scores"])
                        auc_truth_all.extend(m["auc_truth"])
                micro_auc_val = roc_auc_from_scores(np.concatenate(auc_truth_all), np.concatenate(auc_scores_all)) if auc_scores_all else float('nan')
            mic = dict(
                acc = safe_div(tp+tn, tp+fp+fn+tn),
                precision = safe_div(tp, tp+fp),
                recall = safe_div(tp, tp+fn),
                dice = safe_div(2*tp, 2*tp+fp+fn),
                iou = safe_div(tp, tp+fp+fn),
                ASSD=float('nan'), HD95=float('nan'),  # not meaningful for micro
                AUC=micro_auc_val 
            )

            summary_rows.append(dict(
                model=model_tag, class_name=cls,
                macro_acc=mac["acc"], macro_precision=mac["precision"], macro_recall=mac["recall"],
                macro_dice=mac["dice"], macro_iou=mac["iou"], macro_ASSD=mac["ASSD"], macro_HD95=mac["HD95"],macro_auc=mac["AUC"],  
                micro_acc=mic["acc"], micro_precision=mic["precision"], micro_recall=mic["recall"],
                micro_dice=mic["dice"], micro_iou=mic["iou"], micro_auc=mic["AUC"]
            ))

    sum_csv = outdir / "metrics_summary.csv"
    with open(sum_csv, "w", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else
                           ["model","class_name","macro_acc","macro_precision","macro_recall","macro_dice","macro_iou","macro_ASSD","macro_HD95","macro_auc","micro_acc","micro_precision","micro_recall","micro_dice","micro_iou","micro_auc"])
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    print(f"\nWrote:\n  {per_csv}\n  {sum_csv}")
    if _DIST_IMPL == 'none':
        print("NOTE: ASSD/HD95 set to NaN (scipy/skimage not found). Install scipy to enable.")


if __name__ == "__main__":
    main()
