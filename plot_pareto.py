"""
PLOT PARETO GRAPHS FROM CSV
============================

Reads the saved Pareto results CSV and generates the 6-panel visualization.

Usage:
    python plot_pareto_from_csv.py

Input:
    ./pareto/all_runs.csv  (must contain 'pareto' column and objective columns)

Output:
    ./pareto/pareto_graph_from_csv.png
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os


def plot_pareto_graph(csv_path, output_path="./pareto/pareto_graph_from_csv.png"):
    """
    Read CSV and generate 6-panel Pareto visualization.
    
    Args:
        csv_path: Path to the CSV file with Pareto results
        output_path: Where to save the PNG graph
    """
    
    # Read CSV
    if not os.path.exists(csv_path):
        print(f"❌ Error: CSV file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} results from {csv_path}")
    
    # Check required columns exist
    required_cols = ["waiting", "queue", "speed", "pressure_abs", "ttc_conflicts", "queue_length", "pareto"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ Error: Missing columns in CSV: {missing}")
        return
    
    # Extract training parameters if they exist (for subtitle)
    n_agents = len(df)
    n_pareto = df["pareto"].sum()
    
    # Try to infer training steps from CSV if there's a 'run' column pattern
    # Otherwise just show agent count
    if "run" in df.columns:
        subtitle = f"Total candidates: {n_agents}   |   Pareto-optimal: {n_pareto}   |   Training results"
    else:
        subtitle = f"Total candidates: {n_agents}   |   Pareto-optimal: {n_pareto}"
    
    # Create figure
    fig = plt.figure(figsize=(16, 12), facecolor="white")
    fig.suptitle("Pareto Multi-Objective RL — Work Zone Traffic Control",
                 fontsize=18, color="black", fontweight="bold", y=0.97)
    
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3,
                           left=0.07, right=0.96, top=0.90, bottom=0.07)
    
    pairs = [
        ("waiting",        "speed",          "Waiting Time (s)",    "Avg Speed (m/s)"),
        ("waiting",        "queue_length",   "Waiting Time (s)",    "Work Zone Queue (m)"),
        ("waiting",        "ttc_conflicts",  "Waiting Time (s)",    "TTC Conflicts"),
        ("queue",          "speed",          "Queue Count",         "Avg Speed (m/s)"),
        ("ttc_conflicts",  "speed",          "TTC Conflicts",       "Avg Speed (m/s)"),
        ("queue_length",   "ttc_conflicts",  "Work Zone Queue (m)", "TTC Conflicts"),
    ]
    
    for idx, (xkey, ykey, xlabel, ylabel) in enumerate(pairs):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        ax.set_facecolor("white")
        ax.tick_params(colors="black", labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_color("#cccccc")
        
        mask_off = ~df["pareto"]
        mask_on  =  df["pareto"]
        
        # Non-Pareto points (blue)
        ax.scatter(df.loc[mask_off, xkey], df.loc[mask_off, ykey],
                   c="#5555aa", s=60, edgecolors="#3333aa", linewidths=0.8,
                   alpha=0.7, label="Non-Pareto", zorder=2)
        
        # Pareto-optimal points (red)
        ax.scatter(df.loc[mask_on, xkey], df.loc[mask_on, ykey],
                   c="#ff6b6b", s=100, edgecolors="black", linewidths=1.5,
                   alpha=1.0, label="Pareto-Optimal", zorder=3)
        
        ax.set_xlabel(xlabel, color="black", fontsize=9)
        ax.set_ylabel(ylabel, color="black", fontsize=9)
        ax.set_title(f"{xlabel} vs {ylabel}", color="black", fontsize=10, fontweight="bold", pad=8)
        ax.grid(True, color="#dddddd", alpha=0.6)
    
    # Legend
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               fontsize=11, frameon=True, facecolor="white",
               edgecolor="#cccccc", labelcolor="black",
               bbox_to_anchor=(0.5, 0.005))
    
    # Subtitle with stats
    fig.text(0.5, 0.935, subtitle,
             ha="center", fontsize=10, color="black", style="italic")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"📈 Graph saved: {output_path}")
    
    # Optional: show plot interactively
    # plt.show()


if __name__ == "__main__":
    # Default paths
    csv_input = "./pareto/all_runs_adjusted.csv"
    png_output = "./pareto/pareto_graph_from_csv.png"
    
    print("=" * 70)
    print("  PARETO VISUALIZATION FROM CSV")
    print("=" * 70)
    
    plot_pareto_graph(csv_input, png_output)
    
    print("\n✅ Done.")