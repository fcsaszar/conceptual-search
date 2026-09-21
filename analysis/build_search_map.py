#!/usr/bin/env python3
"""Search map: every generated concept positioned by semantic distance
(component-level distance to the nearest seed) and performance (win rate
against the thirty-venture reference set).  Writes figs/search-map.{pdf,png}.

Usage: python3 analysis/build_search_map.py --run-id <id>
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

C_LOCAL, C_GLOBAL, C_SEED = "#b65f45", "#0f766e", "#8a8a8a"
C_LOCAL_LT, C_GLOBAL_LT = "#e8c4b8", "#5f9a92"
INK, INK2 = "#1a1a1a", "#555555"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9.5,
    "axes.edgecolor": "#3c3c3c", "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": "#e8e8e8", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.facecolor": "white", "pdf.fonttype": 42,
})

sq = lambda s: " ".join(str(s).split())



def championship_overrides(BASE, run_id):
    """text-key -> championship-restricted win rate, for plans in both data
    sources (the championship is canonical; bipartite scores fill the rest)."""
    import csv as _csv, json as _json, re as _re
    _csv.field_size_limit(10**9)
    champ = _json.load(open(BASE / "analysis" / "universe-champ-wr.json"))
    sq_ = lambda s: " ".join(str(s).split())
    out = {}
    ent = BASE / "out" / f"{run_id}-polish-entrants.csv"
    if ent.exists():
        by_id = {}
        for r in _csv.DictReader(open(ent)):
            by_id[r["id"]] = sq_(r["plan_text"])
        for cid, v in champ.items():
            m = _re.match(r"(g12-s\d+)-(parent|polished)", cid)
            if m:
                key = f"{m.group(1)}-{m.group(2)}"
                for eid, text in by_id.items():
                    if eid.startswith(key):
                        out[text] = v["universe_wr"]
    lc = BASE / "out" / f"{run_id}-local-candidates.csv"
    if lc.exists():
        sel = [r for r in _csv.DictReader(open(lc)) if r["selected"] == "True"]
        if sel:
            lf = [v for v in champ.values() if v["kind"] == "evolved-local"]
            if lf:
                out[sq_(sel[-1]["plan_text"])] = lf[0]["universe_wr"]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--base-dir", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--seed-id", default="clockchain")
    ap.add_argument("--feature-slot", default=None,
                    help="Ring and label this final-population slot as the "
                         "global counterpart of the local final.")
    args = ap.parse_args()
    BASE = Path(args.base_dir)
    OUT = BASE / "out"
    RID = args.run_id

    lc = pd.read_csv(OUT / f"{RID}-local-candidates.csv")
    gc = pd.read_csv(OUT / f"{RID}-global-candidates.csv")
    uq = pd.read_csv(OUT / f"{RID}-universe-common-quality.csv")
    ss = pd.read_csv(OUT / f"{RID}-semantic-scores.csv")
    champ = json.load(open(BASE / "analysis" / "universe-champ-wr.json"))

    for df in (lc, gc):
        df["key"] = df["plan_text"].map(sq)
    U = dict(zip(uq["text_key"].map(sq), uq["benchmark_quality"]))
    U.update(championship_overrides(BASE, RID))
    N = dict(zip(ss["text_key"].map(sq), ss["semantic_novelty"]))
    # seeds are originals: their bipartite score counts two self-matches as
    # ties, so use the championship-restricted rate instead
    orig_wr = {k: v["universe_wr"] for k, v in champ.items()
               if v["kind"] == "original"}
    gc["otype"] = gc["origin"].str.extract(r"^(seed|elite|mutant|crossover)")

    seeds = gc[gc["step"] == 0][["candidate_id", "key"]].copy()
    seeds["q"] = seeds["candidate_id"].map(orig_wr)

    def states(df, sel_col="selected"):
        path = df[df[sel_col] == True].sort_values("step").copy()
        rows, last = [], None
        for _, r in path.iterrows():
            if r["key"] != last:
                rows.append(r)
                last = r["key"]
        return pd.DataFrame(rows)

    lstates = states(lc)
    lstates["q"] = [orig_wr[args.seed_id] if s == 0 else U.get(k)
                    for s, k in zip(lstates["step"], lstates["key"])]
    lstates["nov"] = lstates["key"].map(N)
    gpath = gc[gc["rank"] == 1].sort_values("step").copy()
    gpath["q"] = [orig_wr.get(cid, U.get(k)) if s == 0 else U.get(k)
                  for s, cid, k in zip(gpath["step"], gpath["candidate_id"],
                                       gpath["key"])]
    gpath["nov"] = gpath["key"].map(N)

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    gen_g = gc[gc["otype"].isin(["mutant", "crossover"])].drop_duplicates("key")
    gen_l = lc[lc["origin"] == "local_variant"].drop_duplicates("key")
    ax.scatter(gen_g["key"].map(N), gen_g["key"].map(U), s=16, marker="D",
               color=C_GLOBAL_LT, edgecolors="none", zorder=2)
    ax.scatter(gen_l["key"].map(N), gen_l["key"].map(U), s=16, marker="o",
               color=C_LOCAL_LT, edgecolors="none", zorder=2)
    ax.axvline(0, color="#cfcfcf", linewidth=0.7, zorder=1)
    label_rows = sorted(
        ((qv, ", ".join(grp["candidate_id"]), len(grp))
         for qv, grp in seeds.dropna(subset=["q"]).groupby(seeds["q"].round(3))),
        key=lambda t: t[0])
    last_y = -1.0
    for qv, names, k in label_rows:
        ax.scatter([0] * k, [qv] * k, marker="^", s=34,
                   facecolors="white", edgecolors=C_SEED, linewidths=1.0, zorder=4)
        ty = max(qv, last_y + 0.036)   # push labels apart when ticks are close
        while any(abs(ty - t) < 0.032 for t in (0, 0.25, 0.5, 0.75, 1.0)):
            ty += 0.034                # and keep clear of the y-tick numbers
        last_y = ty
        # text right-aligned at x=-0.014; the leader starts at the point a
        # \cdot would occupy after the last letter (east midpoint of label)
        ax.text(-0.014, ty, names, ha="right", va="center",
                fontsize=8, color=INK2, zorder=4)
        ax.plot([-0.0115, -0.0025], [ty, qv], color="#c4c4c4", lw=0.6,
                zorder=3, solid_capstyle="round")
    for df, color, ls in ((lstates, C_LOCAL, "-"),
                          (gpath.dropna(subset=["q"]), C_GLOBAL, (0, (4, 2)))):
        ax.plot(df["nov"], df["q"], color=color, linestyle=ls, linewidth=1.5, zorder=5)
        ax.scatter(df["nov"], df["q"], s=24, color=color, zorder=6,
                   edgecolors="white", linewidths=0.6)
    lf = lstates.iloc[-1]
    ax.annotate("local final", (lf["nov"], lf["q"]), xytext=(10, 2),
                textcoords="offset points", va="center", fontsize=8.5,
                color=C_LOCAL, fontweight="bold", zorder=8)
    if args.feature_slot:
        import csv as _csv
        _csv.field_size_limit(10**9)
        fcid = None
        for row in _csv.DictReader(open(OUT / f"{RID}-polish-map.csv")):
            if row["seed_id"] == args.feature_slot:
                fcid = row["candidate_id"]
        frow = gc[gc["candidate_id"] == fcid].iloc[0]
        fx, fy = N.get(frow["key"]), U.get(frow["key"])
        ax.scatter([fx], [fy], s=120, facecolors="none", edgecolors=INK,
                   linewidths=1.4, zorder=7)
        ax.annotate("best of final generation", (fx, fy), xytext=(10, -14),
                    textcoords="offset points", fontsize=8.5, color=C_GLOBAL,
                    fontweight="bold", zorder=8)

    ax.set_xlim(-0.105, max(0.16, gen_g["key"].map(N).max() * 1.15))
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Semantic distance to nearest seed  (component-level)")
    ax.set_ylabel("Performance  (win rate vs. the reference set)")
    handles = [
        Line2D([], [], color=C_LOCAL, marker="o", markersize=4.5, lw=1.5,
               label="Local search path"),
        Line2D([], [], color=C_GLOBAL, marker="o", markersize=4.5, lw=1.5,
               linestyle=(0, (4, 2)), label="Global generation winners"),
        Line2D([], [], color=C_LOCAL_LT, marker="o", markersize=4.5, lw=0,
               label="Variants (local)"),
        Line2D([], [], color=C_GLOBAL_LT, marker="D", markersize=4.5, lw=0,
               label="Variants (global)"),
        Line2D([], [], color=C_SEED, marker="^", markersize=5.5, lw=0,
               markerfacecolor="white", label="Seeds"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    (BASE / "figs").mkdir(exist_ok=True)
    fig.savefig(BASE / "figs" / "search-map.pdf")
    fig.savefig(BASE / "figs" / "search-map.png", dpi=170)
    print("wrote search-map")


if __name__ == "__main__":
    main()
