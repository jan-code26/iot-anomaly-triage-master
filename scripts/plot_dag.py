"""
Generate the causal DAG figure for the paper.

DAG structure (from backend/services/causal_scorer.py):

    op_setting_1 (Altitude)  →  AirDensity → CoolingEfficiency → sensor_4
    op_setting_2 (Mach)      →  TipSpeed   → HPCLoading        → sensor_11
                                                                → sensor_15
    op_setting_3 (TRA)       →  FuelFlow   → CombustorTemp     → sensor_3
                                                                → sensor_9

Outputs:
    paper/causal_dag.png   (for manuscript and README)
    paper/causal_dag.pdf   (for LaTeX submission)

Usage:
    python scripts/plot_dag.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "paper"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------

EDGES = [
    # Branch 1 — Altitude
    ("op_setting_1\n(Altitude)", "AirDensity"),
    ("AirDensity", "CoolingEfficiency"),
    ("CoolingEfficiency", "sensor_4\n(HPC outlet temp)"),
    # Branch 2 — Mach
    ("op_setting_2\n(Mach)", "TipSpeed"),
    ("TipSpeed", "HPCLoading"),
    ("HPCLoading", "sensor_11\n(HPC outlet temp)"),
    ("HPCLoading", "sensor_15\n(HPC outlet pressure)"),
    # Branch 3 — TRA
    ("op_setting_3\n(TRA)", "FuelFlow"),
    ("FuelFlow", "CombustorTemp"),
    ("CombustorTemp", "sensor_3\n(fan inlet temp)"),
    ("CombustorTemp", "sensor_9\n(fan speed)"),
]

# Node categories for coloring
ROOT_NODES = {
    "op_setting_1\n(Altitude)",
    "op_setting_2\n(Mach)",
    "op_setting_3\n(TRA)",
}
LATENT_NODES = {
    "AirDensity",
    "CoolingEfficiency",
    "TipSpeed",
    "HPCLoading",
    "FuelFlow",
    "CombustorTemp",
}
SENSOR_NODES = {
    "sensor_4\n(HPC outlet temp)",
    "sensor_11\n(HPC outlet temp)",
    "sensor_15\n(HPC outlet pressure)",
    "sensor_3\n(fan inlet temp)",
    "sensor_9\n(fan speed)",
}

# ---------------------------------------------------------------------------
# Layout — manual x/y positions so the three branches are visually separated
# ---------------------------------------------------------------------------

POS: dict[str, tuple[float, float]] = {
    # Root nodes — left column
    "op_setting_1\n(Altitude)": (0, 2),
    "op_setting_2\n(Mach)":     (0, 0),
    "op_setting_3\n(TRA)":      (0, -2),

    # Latent nodes — middle two columns
    "AirDensity":        (2, 2),
    "CoolingEfficiency": (4, 2),

    "TipSpeed":   (2,  0.4),
    "HPCLoading": (4,  0.4),

    "FuelFlow":      (2, -2),
    "CombustorTemp": (4, -2),

    # Sensor nodes — right column
    "sensor_4\n(HPC outlet temp)":      (6,  2),
    "sensor_11\n(HPC outlet temp)":     (6,  0.8),
    "sensor_15\n(HPC outlet pressure)": (6, -0.0),
    "sensor_3\n(fan inlet temp)":       (6, -1.5),
    "sensor_9\n(fan speed)":            (6, -2.5),
}

# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------

def main() -> None:
    G = nx.DiGraph()
    G.add_edges_from(EDGES)

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(-0.8, 7.5)
    ax.set_ylim(-3.5, 3.0)
    ax.axis("off")

    # Node colours
    node_colors = []
    for n in G.nodes():
        if n in ROOT_NODES:
            node_colors.append("#4C72B0")   # blue — observable op setting
        elif n in LATENT_NODES:
            node_colors.append("#DD8452")   # orange — latent physical variable
        else:
            node_colors.append("#55A868")   # green — observed sensor

    nx.draw_networkx_nodes(
        G, POS,
        node_color=node_colors,
        node_size=2200,
        alpha=0.92,
        ax=ax,
    )
    nx.draw_networkx_labels(
        G, POS,
        font_size=7.5,
        font_color="white",
        font_weight="bold",
        ax=ax,
    )
    nx.draw_networkx_edges(
        G, POS,
        arrowsize=18,
        arrowstyle="-|>",
        edge_color="#444444",
        width=1.6,
        connectionstyle="arc3,rad=0.05",
        ax=ax,
        min_source_margin=28,
        min_target_margin=28,
    )

    # Column headers
    for x, label in [(0, "Operational\nSettings"), (3, "Latent Physical\nVariables"), (6, "Observed\nSensors")]:
        ax.text(x, 2.75, label, ha="center", va="bottom", fontsize=9,
                fontweight="bold", color="#333333")

    # Legend
    legend_handles = [
        mpatches.Patch(color="#4C72B0", label="Op setting (root node)"),
        mpatches.Patch(color="#DD8452", label="Latent physical variable"),
        mpatches.Patch(color="#55A868", label="Observed sensor"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8.5,
              framealpha=0.9, edgecolor="#cccccc")

    ax.set_title(
        "Causal DAG: Operating Conditions → Latent Variables → Sensor Readings",
        fontsize=11, pad=10,
    )

    fig.tight_layout()

    png_out = OUT_DIR / "causal_dag.png"
    pdf_out = OUT_DIR / "causal_dag.pdf"
    fig.savefig(png_out, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_out, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_out}")
    print(f"Saved: {pdf_out}")


if __name__ == "__main__":
    main()
