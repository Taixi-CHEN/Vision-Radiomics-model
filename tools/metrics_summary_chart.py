#!/usr/bin/env python3
"""
Make comparison charts & a slide-ready table from your metrics CSV.

Usage:
  python make_compare_charts.py \
    --csv /path/to/metrics_summary.csv \
    --outdir biomedparse_charts

What it does:
  • Saves bar charts comparing Base vs Finetuned for:
      - Macro Dice   -> macro_dice_comparison.png
      - Macro IoU    -> macro_iou_comparison.png
      - Micro Dice   -> micro_dice_comparison.png
  • Exports a slide-ready table (4-decimal precision):
      - slide_table.csv
      - slide_table.md (markdown for your PPT/notes)
"""

# import argparse
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# def ensure_class_order(df):
#     # Prefer this order if present; otherwise fall back to natural order with ALL at the end
#     preferred = ["tumor", "stroma", "normal", "ALL"]
#     classes = [c for c in preferred if c in df["class_name"].unique().tolist()]
#     for c in df["class_name"].unique():
#         if c not in classes:
#             classes.append(c)
#     # ensure ALL is last if present
#     if "ALL" in classes:
#         classes = [c for c in classes if c != "ALL"] + ["ALL"]
#     return classes


# def plot_metric(pivot, title, out_path, ylim=(0, 1), dpi=200):
#     """pivot: index=class_name, columns=model, values=metric"""
#     fig, ax = plt.subplots(figsize=(8, 4.2))
#     idx = np.arange(len(pivot.index))
#     width = 0.38

#     base_vals = pivot.get("base", pd.Series(index=pivot.index, dtype=float)).values
#     finetuned_vals = pivot.get("finetuned", pd.Series(index=pivot.index, dtype=float)).values

#     ax.bar(idx - width/2, base_vals, width, label="Base")
#     ax.bar(idx + width/2, finetuned_vals, width, label="Finetuned")

#     ax.set_xticks(idx)
#     ax.set_xticklabels(pivot.index, rotation=0, fontsize=10)
#     ax.set_ylim(*ylim)
#     ax.set_ylabel(title.split(" (", 1)[0])
#     ax.set_title(title, fontsize=12)
#     ax.legend(frameon=False)
#     ax.grid(axis="y", alpha=0.2)

#     fig.tight_layout()
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(out_path, dpi=dpi)
#     plt.close(fig)


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--csv", required=True, help="Path to metrics CSV (your file above).")
#     ap.add_argument("--outdir", default="biomedparse_charts", help="Output folder for charts & tables.")
#     ap.add_argument("--dpi", type=int, default=200)
#     args = ap.parse_args()

#     outdir = Path(args.outdir)
#     outdir.mkdir(parents=True, exist_ok=True)

#     # Load
#     df = pd.read_csv(args.csv)

#     # Keep only the rows we need (base & finetuned)
#     df = df[df["model"].isin(["base", "finetuned"])].copy()

#     # Decide class order
#     class_order = ensure_class_order(df)

#     # -------- Slide-ready table (4-decimal precision) --------
#     table_cols = [
#         "model", "class_name",
#         "macro_acc", "macro_precision", "macro_recall", "macro_dice", "macro_iou",
#         "micro_acc", "micro_precision", "micro_recall", "micro_dice", "micro_iou",
#     ]
#     table_df = df[table_cols].copy()
#     table_df = table_df.sort_values(by=["class_name", "model"], key=lambda s: s.map({c: i for i, c in enumerate(class_order)}).fillna(999))

#     # round to 4 decimals for readability
#     numeric_cols = [c for c in table_df.columns if c not in ["model", "class_name"]]
#     table_df[numeric_cols] = table_df[numeric_cols].astype(float).round(4)

#     # Save CSV + Markdown
#     table_csv = outdir / "slide_table.csv"
#     table_md = outdir / "slide_table.md"
#     table_df.to_csv(table_csv, index=False)
#     with open(table_md, "w") as f:
#         f.write(table_df.to_markdown(index=False))
#     print(f"Saved table:\n  {table_csv}\n  {table_md}")

#     # -------- Charts --------
#     # Macro Dice
#     pivot_macro_dice = (
#         df.pivot_table(index="class_name", columns="model", values="macro_dice")
#           .reindex(class_order)
#     )
#     plot_metric(
#         pivot_macro_dice,
#         title="Macro Dice (Base vs Finetuned)",
#         out_path=outdir / "macro_dice_comparison.png",
#         ylim=(0, 1),
#         dpi=args.dpi,
#     )

#     # Macro IoU
#     pivot_macro_iou = (
#         df.pivot_table(index="class_name", columns="model", values="macro_iou")
#           .reindex(class_order)
#     )
#     plot_metric(
#         pivot_macro_iou,
#         title="Macro IoU (Base vs Finetuned)",
#         out_path=outdir / "macro_iou_comparison.png",
#         ylim=(0, 1),
#         dpi=args.dpi,
#     )

#     # Micro Dice (often preferred when classes have very different sizes)
#     pivot_micro_dice = (
#         df.pivot_table(index="class_name", columns="model", values="micro_dice")
#           .reindex(class_order)
#     )
#     plot_metric(
#         pivot_micro_dice,
#         title="Micro Dice (Base vs Finetuned)",
#         out_path=outdir / "micro_dice_comparison.png",
#         ylim=(0, 1),
#         dpi=args.dpi,
#     )

#     print("Saved charts:")
#     print(f"  {outdir/'macro_dice_comparison.png'}")
#     print(f"  {outdir/'macro_iou_comparison.png'}")
#     print(f"  {outdir/'micro_dice_comparison.png'}")


# if __name__ == "__main__":
#     main()




#!/usr/bin/env python3
# tools/metrics_summary_chart.py
"""
Line charts (Base vs Finetuned) for MACRO metrics only.

Usage:
  python tools/metrics_summary_chart.py \
    --csv /path/to/metrics_summary.csv \
    --outdir charts_out \
    --metrics macro_dice macro_iou macro_precision macro_recall macro_acc macro_ASSD macro_HD95 \
    --annotate
"""

# import argparse
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# BOUNDED_01 = {
#     "macro_acc","macro_precision","macro_recall","macro_dice","macro_iou",
#     "micro_acc","micro_precision","micro_recall","micro_dice","micro_iou"
# }
# DISTANCE_METRICS = {"macro_assd","macro_hd95","micro_assd","micro_hd95"}

# def ensure_class_order(df):
#     preferred = ["tumor", "stroma", "normal", "ALL"]
#     classes = [c for c in preferred if c in df["class_name"].unique().tolist()]
#     for c in df["class_name"].unique():
#         if c not in classes:
#             classes.append(c)
#     if "ALL" in classes:
#         classes = [c for c in classes if c != "ALL"] + ["ALL"]
#     return classes

# def plot_lines(pivot, metric_name, out_path, annotate=False, dpi=220):
#     x = np.arange(len(pivot.index))
#     fig, ax = plt.subplots(figsize=(8, 4))

#     ax.plot(x, pivot.get("base", pd.Series(index=pivot.index, dtype=float)).values,
#             marker="o", linewidth=2, label="Base")
#     ax.plot(x, pivot.get("finetuned", pd.Series(index=pivot.index, dtype=float)).values,
#             marker="s", linewidth=2, label="Finetuned")

#     ax.set_xticks(x)
#     ax.set_xticklabels(pivot.index, rotation=0)

#     mlow = metric_name.lower()
#     if mlow in BOUNDED_01:
#         ax.set_ylim(0, 1)
#         ax.set_ylabel(metric_name.replace("_", " ").title())
#     else:
#         # distance metrics: autoscale with a little headroom
#         vals = pivot.values[np.isfinite(pivot.values)]
#         if vals.size:
#             ymax = float(np.nanmax(vals)) * 1.1 if np.nanmax(vals) > 0 else 1.0
#             ax.set_ylim(0, ymax)
#         ax.set_ylabel(metric_name.replace("Macro_", " ").title() + " (px)")

#     ax.set_title(f"{metric_name.replace('Macro_',' ').title()} — Base vs Finetuned")
#     ax.grid(axis="y", alpha=0.25)
#     ax.legend(frameon=False)

#     if annotate:
#         for i, cls in enumerate(pivot.index):
#             for model in ("base", "finetuned"):
#                 if model in pivot.columns:
#                     val = pivot.loc[cls, model]
#                     if pd.notna(val):
#                         ax.text(i, val + (0.02 if mlow in BOUNDED_01 else 0.02*ax.get_ylim()[1]),
#                                 f"{val:.4f}", ha="center", va="bottom", fontsize=8)

#     fig.tight_layout()
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(out_path, dpi=dpi)
#     plt.close(fig)

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--csv", required=True, help="Path to metrics_summary.csv")
#     ap.add_argument("--outdir", default="charts_out", help="Where to save charts")
#     ap.add_argument("--dpi", type=int, default=220)
#     ap.add_argument("--metrics", nargs="+",
#                     default=["macro_dice", "macro_iou"],
#                     help="Macro metrics to plot as lines")
#     ap.add_argument("--annotate", action="store_true", help="Write values above points")
#     args = ap.parse_args()

#     outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

#     df = pd.read_csv(args.csv)
#     df = df[df["model"].isin(["base", "finetuned"])].copy()
#     if "class_name" not in df.columns and "class" in df.columns:
#         df = df.rename(columns={"class": "class_name"})

#     class_order = ensure_class_order(df)

#     for metric in args.metrics:
#         if metric not in df.columns:
#             print(f"[skip] {metric} not in CSV.")
#             continue
#         pivot = (df.pivot_table(index="class_name", columns="model", values=metric)
#                    .reindex(class_order))
#         out_path = outdir / f"{metric}_linechart.png"
#         plot_lines(pivot, metric, out_path, annotate=args.annotate, dpi=args.dpi)
#         print(f"Saved: {out_path}")

# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
# tools/macro_split_charts.py
# Make two charts from metrics_summary.csv:
#   1) BOUNDED metrics (0–1): precision, recall, dice, iou (+ optional acc)
#   2) DISTANCE metrics (px): ASSD, HD95
# Labels are shown WITHOUT the "macro_" prefix for clean legends.

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Defaults for the two figures
BOUNDED_DEFAULT = ["macro_precision", "macro_recall", "macro_dice", "macro_iou"]
DISTANCE_METRICS = ["macro_ASSD", "macro_HD95"]

# Visuals
COLOR = {
    "macro_dice":      "#1f77b4",
    "macro_iou":       "#ff7f0e",
    "macro_precision": "#2ca02c",
    "macro_recall":    "#d62728",
    "macro_acc":       "#9467bd",
    "macro_ASSD":      "#8c564b",
    "macro_HD95":      "#e377c2",
}
STYLE = {"base": "--", "finetuned": "-"}
MARKER = {"base": "o", "finetuned": "s"}

def class_order(df):
    pref = ["tumor", "stroma", "normal", "ALL"]
    present = df["class_name"].unique().tolist()
    ordered = [c for c in pref if c in present]
    ordered += [c for c in present if c not in ordered and c != "ALL"]
    if "ALL" in present and "ALL" not in ordered:
        ordered.append("ALL")
    return ordered

def clean_label(metric_name: str) -> str:
    # Show labels without "macro_" prefix
    return metric_name.replace("macro_", "")

def plot_group(df, metrics, out_path, title, y_label,
               y_lim=None, annotate="last", figw=12.5, figh=6.2, dpi=220):
    """Plot one group of metrics (all on single y-axis)."""
    order = class_order(df)
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(figw, figh))

    for metric in metrics:
        if metric not in df.columns:
            continue
        piv = (df.pivot_table(index="class_name", columns="model", values=metric)
                 .reindex(order))

        for model in ("base", "finetuned"):
            if model not in piv.columns:
                continue
            y = piv[model].values
            ax.plot(
                x, y,
                linestyle=STYLE.get(model, "-"),
                marker=MARKER.get(model, "o"),
                linewidth=2.2, markersize=6,
                label=f"{clean_label(metric)} ({model})",
                color=COLOR.get(metric)
            )
            # annotations
            if annotate != "none":
                idxs = range(len(y)) if annotate == "all" else [len(y)-1]
                for i in idxs:
                    if i >= 0 and i < len(y) and not pd.isna(y[i]):
                        ax.text(
                            i, y[i] + (0.02 if y_lim is None else 0),
                            f"{y[i]:.4f}" if y_label != "Distance (px)" else f"{y[i]:.2f}",
                            ha="center", va="bottom", fontsize=8
                        )

    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel(y_label)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(title)

    # Legend outside
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(top=0.86, right=0.88)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Split charts: bounded metrics & distance metrics (Base vs Finetuned).")
    ap.add_argument("--csv", required=True, help="metrics_summary.csv")
    ap.add_argument("--outdir", required=True, help="folder to save figures")
    ap.add_argument("--basename", default="macro_split", help="base filename for outputs")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--figw", type=float, default=13.0)
    ap.add_argument("--figh", type=float, default=6.2)
    ap.add_argument("--include_acc", action="store_true", help="Also plot macro_acc in bounded chart")
    ap.add_argument("--only_all", action="store_true", help="Plot only the ALL class row")
    ap.add_argument("--annotate", choices=["none","last","all"], default="last")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "class_name" not in df.columns and "class" in df.columns:
        df = df.rename(columns={"class": "class_name"})
    df = df[df["model"].isin(["base", "finetuned"])].copy()

    if args.only_all:
        df = df[df["class_name"] == "ALL"]

    # Prepare metric lists
    bounded = list(BOUNDED_DEFAULT)
    if args.include_acc and "macro_acc" in df.columns:
        bounded = ["macro_acc"] + bounded
    bounded = [m for m in bounded if m in df.columns]
    distances = [m for m in DISTANCE_METRICS if m in df.columns]

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) Bounded metrics (0–1)
    if bounded:
        plot_group(
            df, bounded,
            outdir / f"{args.basename}_bounded.png",
            title="Segmentation metrics — Base vs Finetuned",
            y_label="Metrics (0–1)",
            y_lim=(0, 1),
            annotate=args.annotate,
            figw=args.figw, figh=args.figh, dpi=args.dpi
        )
    else:
        print("No bounded metrics found to plot.")

    # 2) Distance metrics (px)
    if distances:
        # auto y-limit from data
        vals = []
        for m in distances:
            piv = df.pivot_table(index="class_name", columns="model", values=m)
            vals.extend(list(piv.values.ravel()))
        vals = np.array([v for v in vals if not pd.isna(v)])
        ymax = float(np.nanmax(vals)) if vals.size else 1.0

        plot_group(
            df, distances,
            outdir / f"{args.basename}_distance.png",
            title="Distance metrics — Base vs Finetuned",
            y_label="Distance (px)",
            y_lim=(0, ymax * 1.15 if ymax > 0 else 1.0),
            annotate=args.annotate,
            figw=args.figw, figh=args.figh, dpi=args.dpi
        )
    else:
        print("No distance metrics found to plot.")

if __name__ == "__main__":
    main()
