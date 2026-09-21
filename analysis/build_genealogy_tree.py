#!/usr/bin/env python3
"""Winner genealogy as a top-down genealogical tree.

Extracts the full ancestry of the final generation's tournament winner from
the global-candidates CSV, lays it out with graphviz dot (generations as
layers, crossings minimized), and draws it with matplotlib: generation 0 at
the top, the featured concept at the bottom.  Writes figs/genealogy.{pdf,png}.

Usage: python3 analysis/build_genealogy_tree.py --run-id <id>
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

C_SEED, C_MUTANT, C_CROSS = "#8a8a8a", "#1e8e7e", "#0b4f6c"
INK, INK2 = "#1a1a1a", "#555555"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9.5,
    "figure.facecolor": "white", "pdf.fonttype": 42,
})


def load_ancestry(gc_path, feature_cid=None):
    csv.field_size_limit(10**9)
    with open(gc_path, newline="", encoding="utf-8") as f:
        gc = list(csv.DictReader(f))
    byid = {}
    for r in gc:
        byid.setdefault(r["candidate_id"], r)
    if feature_cid is not None:
        winner = byid[feature_cid]
    else:
        winner = next(r for r in gc if r["step"] == "12" and r["rank"] == "1")
    nodes, edges = {}, []
    stack = [winner["candidate_id"]]
    while stack:
        cid = stack.pop()
        if cid in nodes or cid not in byid:
            continue
        r = byid[cid]
        nodes[cid] = r
        for p in (r["parent_a"], r["parent_b"]):
            if p:
                edges.append((p, cid))
                stack.append(p)
    return winner, nodes, edges


def dot_layout(nodes, edges):
    """Layered x-coordinates from graphviz dot (one rank per generation)."""
    alias = {cid: f"n{i}" for i, cid in enumerate(nodes)}
    lines = ["digraph G {", "rankdir=TB;", 'node [shape=circle width=0.28 fixedsize=true label=""];',
             "nodesep=0.18; ranksep=0.42;"]
    by_step = {}
    for cid, r in nodes.items():
        by_step.setdefault(int(r["step"]), []).append(cid)
    for step, cids in sorted(by_step.items()):
        lines.append("{rank=same; " + "; ".join(alias[c] for c in cids) + ";}")
    for p, ch in edges:
        lines.append(f"{alias[p]} -> {alias[ch]};")
    lines.append("}")
    out = subprocess.run(["dot", "-Tplain"], input="\n".join(lines),
                         capture_output=True, text=True, check=True).stdout
    back = {v: k for k, v in alias.items()}
    pos = {}
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0] == "node":
            pos[back[parts[1]]] = float(parts[2])
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--base-dir", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--feature-slot", default=None,
                    help="Feature this final-population slot (e.g. g12-s04) "
                         "instead of the last generation's tournament winner.")
    args = ap.parse_args()
    BASE = Path(args.base_dir)
    gc_path = BASE / "out" / f"{args.run_id}-global-candidates.csv"

    feature_cid = None
    if args.feature_slot:
        pm_path = BASE / "out" / f"{args.run_id}-polish-map.csv"
        for row in csv.DictReader(open(pm_path)):
            if row["seed_id"] == args.feature_slot:
                feature_cid = row["candidate_id"]
                break
        assert feature_cid, f"slot {args.feature_slot} not in polish map"
    winner, nodes, edges = load_ancestry(gc_path, feature_cid)
    xs = dot_layout(nodes, edges)
    xmax = max(xs.values()) or 1.0
    pos = {cid: (xs[cid] / xmax, int(r["step"])) for cid, r in nodes.items()}

    fig, ax = plt.subplots(figsize=(6.9, 4.45))
    for p, ch in edges:
        (x0, y0), (x1, y1) = pos[p], pos[ch]
        is_cross = nodes[ch]["origin"].startswith("crossover")
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=2,
                    arrowprops=dict(arrowstyle="-|>,head_width=0.10,head_length=0.22",
                                    color="#9aa5a3" if is_cross else "#8b8b8b",
                                    lw=0.85,
                                    linestyle=(0, (3, 2)) if is_cross else "-",
                                    shrinkA=3.5, shrinkB=3.5))
    winner_id = winner["candidate_id"]
    for cid, r in nodes.items():
        x, y = pos[cid]
        origin = r["origin"]
        is_cross = origin.startswith("crossover")
        color = (C_SEED if origin.startswith("seed") else
                 C_CROSS if is_cross else C_MUTANT)
        is_winner = cid == winner_id and int(r["step"]) >= 12
        ax.scatter([x], [y], s=150 if is_winner else (58 if is_cross else 52),
                   marker="D" if is_cross else "o", color=color, zorder=5,
                   edgecolors=INK if is_winner else "white",
                   linewidths=1.3 if is_winner else 0.7)
        if origin.startswith("seed"):
            ax.annotate(cid, (x, y), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK,
                        fontweight="bold", annotation_clip=False)
    wx, wy = pos[winner_id]
    FINAL_GEN = 12
    carry = int(winner["step"]) < FINAL_GEN
    # reference-set performance beside every node (bipartite scores; seeds
    # and championship-measured plans use their championship values)
    sq_ = lambda t: " ".join(str(t).split())
    U = {}
    uq_path = BASE / "out" / f"{args.run_id}-universe-common-quality.csv"
    if uq_path.exists():
        for row in csv.DictReader(open(uq_path)):
            U[sq_(row["text_key"])] = float(row["benchmark_quality"])
    import json as _json, re as _re
    seed_wr = {}
    champ_path = BASE / "analysis" / "universe-champ-wr.json"
    if champ_path.exists():
        champ = _json.load(open(champ_path))
        ent_path = BASE / "out" / f"{args.run_id}-polish-entrants.csv"
        if ent_path.exists():
            by_id = {r["id"]: sq_(r["plan_text"])
                     for r in csv.DictReader(open(ent_path))}
            for cid, v in champ.items():
                m = _re.match(r"(g12-s\d+)-parent", cid)
                if m:
                    for eid, text in by_id.items():
                        if eid.startswith(m.group(1) + "-parent"):
                            U[text] = v["universe_wr"]
        seed_wr = {k: v["universe_wr"] for k, v in champ.items()
                   if v["kind"] == "original"}
    fy = FINAL_GEN if carry else wy
    if carry:
        # elitism: the concept enters the final generation unchanged
        ax.plot([wx, wx], [wy + 0.12, FINAL_GEN - 0.14], color="#b0b0b0",
                lw=1.6, zorder=2)
        w_origin = winner["origin"]
        w_color = (C_CROSS if w_origin.startswith("crossover") else C_MUTANT)
        w_marker = "D" if w_origin.startswith("crossover") else "o"
        ax.scatter([wx], [FINAL_GEN], s=150, marker=w_marker, color=w_color,
                   zorder=5, edgecolors=INK, linewidths=1.3)
        w_wr = U.get(sq_(winner["plan_text"]))
        if w_wr is not None:
            ax.annotate(f"{w_wr:.0%}", (wx, FINAL_GEN), xytext=(8, -2),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=7, color=INK2)
    ax.annotate("best of final generation", (wx, fy), xytext=(0, -16),
                textcoords="offset points", ha="center", fontsize=8.5,
                fontweight="bold", color=INK, annotation_clip=False)
    for cid, r in nodes.items():
        x, y = pos[cid]
        wr = (seed_wr.get(cid) if r["origin"].startswith("seed")
              else U.get(sq_(r["plan_text"])))
        if wr is None:
            continue
        ax.annotate(f"{wr:.0%}", (x, y), xytext=(6, -2),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=7, color=INK2)

    ax.set_ylim(12.8, -0.9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_yticks(range(13))
    ax.set_ylabel("Generation")
    ax.set_xticks([])
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#3c3c3c")
    ax.spines["left"].set_linewidth(0.6)
    handles = [
        Line2D([], [], color=C_SEED, marker="o", lw=0, markersize=6, label="Seed venture"),
        Line2D([], [], color=C_MUTANT, marker="o", lw=0, markersize=6, label="Mutation"),
        Line2D([], [], color=C_CROSS, marker="D", lw=0, markersize=6, label="Crossover"),
        Line2D([], [], color="#9aa5a3", lw=1, linestyle=(0, (3, 2)), label="Crossover edge"),
        Line2D([], [], color="#b0b0b0", lw=1.6, label="Elite copy (unchanged)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    (BASE / "figs").mkdir(exist_ok=True)
    fig.savefig(BASE / "figs" / "genealogy.pdf")
    fig.savefig(BASE / "figs" / "genealogy.png", dpi=170)
    n_cross = sum(1 for r in nodes.values() if r["origin"].startswith("crossover"))
    n_seed = sum(1 for r in nodes.values() if r["origin"].startswith("seed"))
    print(f"wrote genealogy ({len(nodes)} nodes: {n_seed} seeds, "
          f"{n_cross} crossovers, {len(nodes)-n_seed-n_cross} mutations)")


if __name__ == "__main__":
    main()
