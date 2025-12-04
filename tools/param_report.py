#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# repo-local imports
from utilities.arguments import load_opt_from_config_files
from modeling.BaseModel import BaseModel
from modeling import build_model

def human(n):
    return f"{n:,}"

def param_rows(named_params):
    rows = []
    for name, p in named_params:
        numel = p.numel()
        rows.append({
            "name": name,
            "shape": tuple(p.shape),
            "dtype": str(p.dtype).replace("torch.", ""),
            "requires_grad": bool(p.requires_grad),
            "numel": numel
        })
    return rows

def group_key(name, depth=1):
    # group by first <depth> segments, e.g. "backbone", "sem_seg_head", "lang_encoder", ...
    parts = name.split(".")
    depth = max(1, min(depth, len(parts)))
    return ".".join(parts[:depth])

def summarize(rows, depth=1):
    # per-group totals
    agg = {}
    for r in rows:
        g = group_key(r["name"], depth)
        d = agg.setdefault(g, {"total":0, "trainable":0})
        d["total"] += r["numel"]
        if r["requires_grad"]:
            d["trainable"] += r["numel"]
    # convert to list with ratio
    out = []
    for g, d in sorted(agg.items(), key=lambda kv: kv[1]["trainable"], reverse=True):
        tr = d["trainable"]
        tot = d["total"]
        out.append({
            "group": g,
            "trainable_params": tr,
            "total_params": tot,
            "trainable_ratio": 0.0 if tot==0 else tr / tot
        })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/biomed_seg_lang_v1.yaml",
                    help="Config file to build the model (training config recommended)")
    ap.add_argument("--ckpt", default="", help="Optional checkpoint .pt to load weights")
    ap.add_argument("--group_depth", type=int, default=1, help="How many name segments to group by")
    ap.add_argument("--outdir", default="param_report_out", help="Where to write CSVs")
    args = ap.parse_args()

    # --- build model ---
    opt = load_opt_from_config_files([args.config])
    import torch
    opt['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'  # <-- ADD THIS
    base = BaseModel(opt, build_model(opt))
    model = base  # BaseModel wrapper (has .model inside)

    if args.ckpt:
        print(f"Loading weights from: {args.ckpt}")
        model = base.from_pretrained(args.ckpt)

    # We want the underlying actual nn.Module to introspect names cleanly.
    nnmod = model.model if hasattr(model, "model") else model

    # --- collect rows ---
    rows = param_rows(nnmod.named_parameters())

    # totals
    total = sum(r["numel"] for r in rows)
    trainable = sum(r["numel"] for r in rows if r["requires_grad"])
    frozen = total - trainable

    # memory info (float32 baseline; training may use AMP/fp16)
    bytes_fp32 = total * 4
    bytes_trainable_fp32 = trainable * 4

    print("\n=== Parameter Summary ===")
    print("Total parameters:    ", human(total))
    print("Trainable parameters:", human(trainable))
    print("Frozen parameters:   ", human(frozen))
    print(f"~Model size (fp32):   {bytes_fp32/1e6:.1f} MB (trainable ~{bytes_trainable_fp32/1e6:.1f} MB)")
    print("\nTop groups by trainable params:")

    group_summary = summarize(rows, depth=args.group_depth)
    for g in group_summary[:15]:
        print(f"  {g['group']:<24} trainable={human(g['trainable_params'])}  "
              f"total={human(g['total_params'])}  ratio={g['trainable_ratio']:.2f}")

    # a few examples of frozen params (if any)
    frozen_examples = [r["name"] for r in rows if not r["requires_grad"]][:10]
    if frozen_examples:
        print("\nExamples frozen params:")
        for n in frozen_examples:
            print(" ", n)

    # --- write CSVs ---
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with (outdir / "all_parameters.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name","shape","dtype","requires_grad","numel"])
        for r in rows:
            w.writerow([r["name"], str(r["shape"]), r["dtype"], int(r["requires_grad"]), r["numel"]])

    with (outdir / "group_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group","trainable_params","total_params","trainable_ratio"])
        for g in group_summary:
            w.writerow([g["group"], g["trainable_params"], g["total_params"], f"{g['trainable_ratio']:.6f}"])

    with (outdir / "totals.txt").open("w") as f:
        f.write(f"total_params={total}\ntrainable_params={trainable}\nfrozen_params={frozen}\n")

    print(f"\nWrote:\n  {outdir/'all_parameters.csv'}\n  {outdir/'group_summary.csv'}\n  {outdir/'totals.txt'}")

if __name__ == "__main__":
    main()
