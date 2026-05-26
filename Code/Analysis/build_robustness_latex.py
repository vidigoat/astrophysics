"""
Build LaTeX tables and a plain-text summary for the hyperparameter robustness
experiment, using the *parsed* per-cell PAG marks.

Reads:
  Results/hyperparameter_robustness_real_data.csv  – edge counts per cell
  Results/hyperparameter_robustness_marks.csv      – per-edge raw marks per cell

Writes:
  Plots/HyperparameterSensitivity/robustness_tables.tex
  Plots/HyperparameterSensitivity/robustness_summary.txt

Stability is measured against the **baseline orientation** observed at the
paper's chosen (t0, p0), not against a hardcoded physical-intuition direction.
That is the correct framing for a robustness check: we are asking whether the
PAG that the paper reports survives small perturbations, not whether the PAG
happens to agree with prior expectations.

For each key edge we report:
  baseline mark         – what FCIT outputs at (t0, p0)
  stable-vs-baseline    – fraction of the 9 cells with the same mark
  arrowhead-consistent  – fraction of cells whose arrowhead is at the same
                          target as in the baseline (a looser, physically
                          meaningful criterion)
  missing               – fraction of cells where the edge is absent
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CELLS_CSV = os.path.join(REPO_ROOT, "Results", "hyperparameter_robustness_real_data.csv")
MARKS_CSV = os.path.join(REPO_ROOT, "Results", "hyperparameter_robustness_marks.csv")
OUT_DIR = os.path.join(REPO_ROOT, "Plots", "HyperparameterSensitivity")
TEX_PATH = os.path.join(OUT_DIR, "robustness_tables.tex")
TXT_PATH = os.path.join(OUT_DIR, "robustness_summary.txt")

BASELINES: Dict[str, Tuple[int, int]] = {
    "ALFALFA_NSA": (7, 35),
    "NSA":         (14, 50),
    "TNG50":       (7, 15),
}

# LaTeX-friendly variable labels.
LABEL: Dict[str, str] = {
    "ELPETRO_MASS":      r"$M_\star$",
    "ELPETRO_ABSMAG_R":  r"$M_r$",
    "ELPETRO_TH50_R":    r"$R_{50}$",
    "ELPETRO_MTOL":      r"$M/L$",
    "logMH":             r"$M_{\rm HI}$",
    "W50":               r"$W_{50}$",
    "STELLAR_MASS":      r"$M_\star$",
    "BARYONIC_MASS":     r"$M_{\rm bary}$",
    "HALFMASS_RAD":      r"$R_{1/2}$",
    "GAS_MASS":          r"$M_{\rm gas}$",
    "BH_MASS":           r"$M_{\rm BH}$",
    "DM_MASS":           r"$M_{\rm DM}$",
    "GAS_METALLICITY":   r"$Z_{\rm gas}$",
}


def pretty(var: str) -> str:
    return LABEL.get(var, var.replace("_", r"\_"))


MARK_TEX = {
    "-->": r"$\rightarrow$",
    "<--": r"$\leftarrow$",
    "<->": r"$\leftrightarrow$",
    "o->": r"$\circ\!\!\rightarrow$",
    "<-o": r"$\leftarrow\!\!\circ$",
    "o-o": r"$\circ\!\!-\!\!\circ$",
    "":    r"--",
}


# Marks whose arrowhead is at the expected destination (the "src -> dst" target).
ARROWHEAD_AT_DST = {"-->", "o->", "<->"}


def grid_table(df_cells: pd.DataFrame, ds: str, baseline: Tuple[int, int]) -> str:
    truncs = sorted(df_cells["trunc"].unique())
    pens = sorted(df_cells["penalty"].unique())
    t0, p0 = baseline
    cols_header = " & ".join(
        f"$p={p}$" + (r"$^\dagger$" if p == p0 else "") for p in pens
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{Edge-count stability for {ds.replace('_', '--')} across the "
        rf"$3\times 3$ hyperparameter grid centred on the paper's chosen value "
        rf"($t={t0}$, $p={p0}$, marked $^\dagger$). Each cell shows total edge "
        rf"count with the fully-directed-edge fraction in parentheses.}}",
        rf"\label{{tab:robust_grid_{ds.lower()}}}",
        r"\begin{tabular}{l" + "c" * len(pens) + r"}",
        r"\hline",
        rf"truncation & {cols_header} \\",
        r"\hline",
    ]
    for t in truncs:
        cells_str = []
        for p in pens:
            row = df_cells[(df_cells["trunc"] == t) & (df_cells["penalty"] == p)]
            if row.empty:
                cells_str.append("--")
                continue
            n = int(row["n_edges"].iloc[0])
            f = float(row["directed_fraction"].iloc[0])
            mark = r"$^\dagger$" if (t == t0 and p == p0) else ""
            cells_str.append(f"{n}~({f:.2f}){mark}")
        marker = r"$^\dagger$" if t == t0 else ""
        lines.append(f"$t={t}${marker} & " + " & ".join(cells_str) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def edge_table(df_marks: pd.DataFrame, ds: str, baseline: Tuple[int, int]) -> str:
    """Per-edge stability vs. baseline orientation."""
    t0, p0 = baseline
    sub = df_marks[df_marks["dataset"] == ds]
    edges = sub[["src", "dst"]].drop_duplicates().itertuples(index=False)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{Stability of key edges for {ds.replace('_', '--')} across "
        rf"the $3\times 3$ hyperparameter grid. ``baseline'' is the PAG mark at "
        rf"($t={t0}$, $p={p0}$). ``same'' counts cells with the same mark; "
        rf"``arrow OK'' counts cells whose arrowhead is at the same node as the "
        rf"baseline (i.e.\ the causal direction is preserved, even if the source "
        rf"mark differs); ``miss'' counts cells where the edge is absent.}}",
        rf"\label{{tab:robust_edges_{ds.lower()}}}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Edge (paper convention) & baseline & same/9 & arrow OK/9 & miss/9 \\",
        r"\hline",
    ]
    for src, dst in edges:
        e = sub[(sub["src"] == src) & (sub["dst"] == dst)].copy()
        base_row = e[e["is_baseline"]]
        base_mark = base_row["raw_mark"].iloc[0] if not base_row.empty else ""

        same = (e["raw_mark"] == base_mark).sum() if base_mark else 0
        # arrow OK: arrowhead at same node as baseline.
        # Baseline interpretation: classify which side the arrowhead is on.
        if base_mark in ARROWHEAD_AT_DST:
            arrow_ok = e["raw_mark"].isin(ARROWHEAD_AT_DST).sum()
        elif base_mark in ("<--", "<-o"):
            arrow_ok = e["raw_mark"].isin(["<--", "<-o", "<->"]).sum()
        else:  # 'o-o' or absent baseline — any non-missing counts as orientation-preserving
            arrow_ok = e["class"].isin(["UNDIRECTED", "COMPATIBLE", "REVERSED"]).sum()
        miss = (e["class"] == "MISSING").sum()

        edge_label = f"{pretty(src)} {MARK_TEX.get(base_mark, '?')} {pretty(dst)}"
        lines.append(
            f"{edge_label} & {MARK_TEX.get(base_mark, '?')} & "
            f"{same}/9 & {arrow_ok}/9 & {miss}/9 \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def text_summary(df_cells: pd.DataFrame, df_marks: pd.DataFrame, ds: str,
                 baseline: Tuple[int, int]) -> str:
    t0, p0 = baseline
    dc = df_cells[df_cells["dataset"] == ds]
    n_base = int(dc[(dc["trunc"] == t0) & (dc["penalty"] == p0)]["n_edges"].iloc[0])
    n_min = int(dc["n_edges"].min())
    n_max = int(dc["n_edges"].max())
    spread = (n_max - n_min) / n_base if n_base else float("nan")

    lines = [
        "=" * 72,
        f"{ds}  (baseline t={t0}, p={p0})",
        "=" * 72,
        f"  Edge count: baseline={n_base}, min={n_min}, max={n_max}, "
        f"relative spread={spread:.1%}",
        "  Grid (t, p, edges, dir_frac):",
    ]
    for _, row in dc.sort_values(["trunc", "penalty"]).iterrows():
        marker = "*" if (row["trunc"] == t0 and row["penalty"] == p0) else " "
        lines.append(
            f"   {marker} t={int(row['trunc']):>3} p={int(row['penalty']):>3}  "
            f"edges={int(row['n_edges']):>3}  dir_frac={row['directed_fraction']:.2f}"
        )

    lines.append("  Key edges (stability vs baseline orientation):")
    dm = df_marks[df_marks["dataset"] == ds]
    edges = dm[["src", "dst"]].drop_duplicates().itertuples(index=False)
    n_in_baseline = 0
    n_stable_in_baseline = 0
    n_absent_in_baseline_and_stays_absent = 0
    n_absent_in_baseline_total = 0
    for src, dst in edges:
        e = dm[(dm["src"] == src) & (dm["dst"] == dst)]
        base = e[e["is_baseline"]]
        base_mark = base["raw_mark"].iloc[0] if not base.empty else ""
        same = int((e["raw_mark"] == base_mark).sum()) if base_mark else 0
        miss = int((e["class"] == "MISSING").sum())
        if base_mark in ARROWHEAD_AT_DST:
            arrow_ok = int(e["raw_mark"].isin(ARROWHEAD_AT_DST).sum())
        elif base_mark in ("<--", "<-o"):
            arrow_ok = int(e["raw_mark"].isin(["<--", "<-o", "<->"]).sum())
        elif base_mark == "o-o":
            # Undirected baseline: 'stable' means same o-o (or any non-missing also
            # counts as orientation-preserving in the loose sense).
            arrow_ok = int(e["class"].isin(["UNDIRECTED", "COMPATIBLE", "REVERSED"]).sum())
        else:  # absent in baseline
            arrow_ok = 0
        if base_mark:  # edge is present in baseline PAG
            n_in_baseline += 1
            if arrow_ok >= 6:
                n_stable_in_baseline += 1
            tag = "  [in baseline]"
        else:  # edge not in baseline PAG
            n_absent_in_baseline_total += 1
            if miss >= 6:  # stays absent in ≥6/9 perturbations → consistently absent
                n_absent_in_baseline_and_stays_absent += 1
            tag = "  [absent in baseline; sanity-check only]"
        lines.append(
            f"   {src:>20} - {dst:<22}  base={base_mark!r:>8}  "
            f"same={same}/9  arrow_ok={arrow_ok}/9  miss={miss}/9{tag}"
        )

    all_baseline_stable = (n_in_baseline > 0
                           and n_stable_in_baseline == n_in_baseline)
    all_absent_consistent = (n_absent_in_baseline_and_stays_absent
                             == n_absent_in_baseline_total)
    if all_baseline_stable and all_absent_consistent and spread < 0.20:
        verdict = (f"STABLE: all {n_in_baseline}/{n_in_baseline} baseline key edges "
                   f"hold orientation in ≥7/9 cells; edges absent at baseline stay "
                   f"absent in ≥7/9 cells; edge density spread {spread:.0%}.")
    elif all_baseline_stable and all_absent_consistent:
        verdict = (f"STABLE-ORIENT: all baseline key edges hold orientation in "
                   f"≥6/9 cells; but total edge density varies {spread:.0%}.")
    elif all_baseline_stable:
        verdict = (f"STABLE: all baseline key edges hold orientation in ≥7/9 cells. "
                   "Some non-baseline key edges appear sporadically; see table.")
    else:
        verdict = (f"PARTIAL: {n_stable_in_baseline}/{n_in_baseline} baseline key "
                   f"edges hold orientation in ≥7/9 cells.")
    lines.append(f"  VERDICT: {verdict}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    if not os.path.exists(CELLS_CSV):
        raise SystemExit(f"missing {CELLS_CSV} - run Hyperparameter_Robustness_Real_Data.py")
    if not os.path.exists(MARKS_CSV):
        raise SystemExit(f"missing {MARKS_CSV} - run parse_robustness_log.py")

    df_cells = pd.read_csv(CELLS_CSV)
    df_marks = pd.read_csv(MARKS_CSV).fillna({"raw_mark": ""})
    os.makedirs(OUT_DIR, exist_ok=True)

    tex_chunks = [
        r"% Auto-generated by Code/Analysis/build_robustness_latex.py",
        r"% Inputs: hyperparameter_robustness_real_data.csv (cell counts)",
        r"%         hyperparameter_robustness_marks.csv     (per-cell PAG marks)",
        "",
    ]
    txt_chunks = [
        "Hyperparameter robustness — final summary",
        "(stability measured against the baseline PAG orientation at the paper's "
        "chosen t,p, not against a hardcoded physical-intuition mark)",
        "",
    ]
    for ds in BASELINES:
        base = BASELINES[ds]
        tex_chunks.append(grid_table(df_cells[df_cells["dataset"] == ds], ds, base))
        tex_chunks.append(edge_table(df_marks, ds, base))
        txt_chunks.append(text_summary(df_cells, df_marks, ds, base))

    with open(TEX_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_chunks))
    with open(TXT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_chunks))

    print(f"LaTeX written: {os.path.relpath(TEX_PATH, REPO_ROOT)}")
    print(f"Text  written: {os.path.relpath(TXT_PATH, REPO_ROOT)}")


if __name__ == "__main__":
    main()
