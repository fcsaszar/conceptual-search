#!/usr/bin/env python3
"""Fig: the polish championship — global discovers, local polishes.

Dumbbell per final-population lineage: open circle = step-12 parent, filled
circle = its polished descendant, both measured as the share of championship
matches won against the 30 original ventures (the reference-set win rate).  The
originals appear as tick rugs on the same axis.  Reads the restricted win
rates from analysis/universe-champ-wr.json (computed from the championship
match records).  Writes figs/polish.{pdf,png} and
figs/polish-facts.json.

Usage: python3 analysis/build_polish_figure.py --run-id <id>
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

C_LOCAL, C_SEED, C_HELD = "#b65f45", "#c9c9c9", "#4a4a4a"
C_PARENT, C_POLISH = "#b78bc4", "#7c4d8f"
INK, INK2 = "#1a1a1a", "#555555"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9.5,
    "axes.edgecolor": "#3c3c3c", "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": "#e8e8e8", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.facecolor": "white", "pdf.fonttype": 42,
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--base-dir", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--firm-set", default="in/firm-set-15-evalrank.txt")
    args = ap.parse_args()
    BASE = Path(args.base_dir)

    uni = json.load(open(BASE / "analysis" / "universe-champ-wr.json"))
    archive = {l.split("#")[0].strip()
               for l in open(BASE / args.firm_set) if l.strip()}

    def seed_of(row_id):
        m = re.match(r"(g12-s\d+)-", str(row_id))
        return m.group(1) if m else None

    parents = {seed_of(k): v for k, v in uni.items() if v["kind"] == "g12-parent"}
    polished = {seed_of(k): v for k, v in uni.items() if v["kind"] == "polished-descendant"}
    lineages = sorted(polished, key=lambda s: -polished[s]["universe_wr"])

    originals = {k: v for k, v in uni.items() if v["kind"] == "original"}
    arch = {k: v["universe_wr"] for k, v in originals.items() if k in archive}
    held = {k: v["universe_wr"] for k, v in originals.items() if k not in archive}
    best2 = sorted(originals.items(), key=lambda kv: -kv[1]["universe_wr"])[:2]

    fig, ax = plt.subplots(figsize=(6.9, 4.1))
    ax.scatter(list(arch.values()), [len(lineages) + 1.0] * len(arch), marker="|",
               s=90, color="#b8b8b8", zorder=3)
    ax.scatter(list(held.values()), [len(lineages) + 1.6] * len(held), marker="|",
               s=90, color=C_HELD, zorder=3)
    for j, (k, v) in enumerate(best2):
        ax.axvline(v["universe_wr"], color="#8a8a8a", lw=0.7, linestyle=(0, (4, 3)), zorder=1)
        ax.annotate(k, (v["universe_wr"], len(lineages) + 2.25 + 0.75 * j),
                    ha="left", xytext=(1, 0), textcoords="offset points",
                    fontsize=8, color=INK2, annotation_clip=False)
    for i, s in enumerate(lineages):
        p, q = parents.get(s), polished[s]
        y = len(lineages) - 1 - i
        if p is not None:
            ax.annotate("", xy=(q["universe_wr"], y), xytext=(p["universe_wr"], y),
                        zorder=4,
                        arrowprops=dict(arrowstyle="-|>,head_width=0.14,head_length=0.28",
                                        color="#b9a6c6", lw=1.1, shrinkA=4, shrinkB=4))
            ax.scatter([p["universe_wr"]], [y], s=34, facecolors="white",
                       edgecolors=C_PARENT, linewidths=1.2, zorder=5)
        ax.scatter([q["universe_wr"]], [y], s=40, color=C_POLISH, zorder=6)
    lf = [v for v in uni.values() if v["kind"] == "evolved-local"]
    if lf:
        ax.axvline(lf[0]["universe_wr"], color=C_LOCAL, lw=0.8, linestyle=(0, (2, 2)))
        ax.annotate("local final", (lf[0]["universe_wr"], -2.4), ha="right",
                    xytext=(-1, 0), textcoords="offset points",
                    fontsize=8, color=C_LOCAL, annotation_clip=False)

    ax.set_yticks(range(len(lineages) + 3))
    labels = ([f"lineage {s[-2:]}" for s in reversed(lineages)]
              + ["", "reference set: bottom 15 (seeds)", "reference set: top 15"])
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_ylim(-3.6, len(lineages) + 2.9)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Performance (win rate vs. the reference set)")
    handles = [
        Line2D([], [], color=C_PARENT, marker="o", lw=0, markersize=6,
               markerfacecolor="white", label="Final-generation concept"),
        Line2D([], [], color=C_POLISH, marker="o", lw=0, markersize=6,
               label="Same concept after local polish"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.01), frameon=False, fontsize=8.5)
    fig.tight_layout()
    (BASE / "figs").mkdir(exist_ok=True)
    fig.savefig(BASE / "figs" / "polish.pdf")
    fig.savefig(BASE / "figs" / "polish.png", dpi=170)
    print("wrote polish")

    deltas = [polished[s]["universe_wr"] - parents[s]["universe_wr"]
              for s in lineages if s in parents]
    owr = [v["universe_wr"] for v in originals.values()]
    best_orig_wr = max(owr)
    facts = {
        "measure": "win rate vs the 30 originals (championship matches, restricted)",
        "median_delta_pp": round(100 * pd.Series(deltas).median(), 1),
        "mean_delta_pp": round(100 * pd.Series(deltas).mean(), 1),
        "n_improved": int(sum(d > 0 for d in deltas)),
        "n_pairs": len(deltas),
        "n_polished_above_best_original": int(sum(polished[s]["universe_wr"] > best_orig_wr
                                                  for s in lineages)),
        "polished_beats_of_30": {s: int(sum(polished[s]["universe_wr"] > w for w in owr))
                                 for s in lineages},
        "median_polished_wr": round(float(pd.Series(
            [polished[s]["universe_wr"] for s in lineages]).median()), 3),
        "median_parent_wr": round(float(pd.Series(
            [parents[s]["universe_wr"] for s in lineages if s in parents]).median()), 3),
        "archive_mean_wr": round(float(pd.Series(list(arch.values())).mean()), 3),
        "held_mean_wr": round(float(pd.Series(list(held.values())).mean()), 3),
    }
    (BASE / "figs" / "polish-facts.json").write_text(json.dumps(facts, indent=2))
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
