"""
plot_results.py — Thesis Comparison Plots: FedAvg vs. FedLMAD
==============================================================
Loads the JSON result files produced by main.py and generates publication-
quality figures for the thesis comparing FedAvg (no defence) against the
proposed FedLMAD strategy under a 30% label-flipping attack.

Usage
-----
    python plot_results.py

Outputs (saved to results/)
---------------------------
    comparison_accuracy.png   — Centralized accuracy over rounds
    comparison_loss.png       — Centralized loss over rounds
    comparison_combined.png   — Both panels side-by-side (main thesis figure)
    lmad_rejections.png       — Per-layer rejection counts over rounds (L-MAD only)
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
FEDAVG_FILE = os.path.join(RESULTS_DIR, "fedavg_results.json")
LMAD_FILE   = os.path.join(RESULTS_DIR, "lmad_results.json")

# Thesis-style colour palette
COLOR_FEDAVG = "#E05A4E"   # red  — baseline (under attack, no defence)
COLOR_LMAD   = "#4477AA"   # blue — proposed defence

LAYER_COLORS = [
    "#4477AA", "#66CCEE", "#228833", "#CCBB44",
    "#EE6677", "#AA3377", "#BBBBBB", "#000000",
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_series(data: list) -> tuple[list, list]:
    """Return (rounds, values) from a list of {round, accuracy/loss} dicts."""
    rounds = [d["round"] for d in data]
    values = [d.get("accuracy", d.get("loss")) for d in data]
    return rounds, values


def _title_info(lmad: dict) -> str:
    """Build a shared subtitle string from lmad metadata."""
    meta = lmad["metadata"]
    K = meta.get("num_clients", "?")
    mal = meta.get("num_malicious", "?")
    pct = round(int(mal) / int(K) * 100) if K != "?" else "?"
    rounds = meta.get("num_rounds", len(lmad["centralized_accuracy"]) - 1)
    return f"{pct}% Label-Flipping Attack, CIFAR-10, K={K}, {rounds} Rounds"


# ──────────────────────────────────────────────────────────────────────────────
# Plot 1 & 2: Accuracy and Loss (separate + combined)
# ──────────────────────────────────────────────────────────────────────────────
def plot_combined(fedavg: dict, lmad: dict) -> None:
    """Side-by-side accuracy and loss curves for both strategies."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"FedAvg vs. FedLMAD under {_title_info(lmad)}",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # ---- Accuracy panel ----
    ax = axes[0]
    r_fa, a_fa = extract_series(fedavg["centralized_accuracy"])
    r_lm, a_lm = extract_series(lmad["centralized_accuracy"])

    ax.plot(r_fa, [v * 100 for v in a_fa],
            color=COLOR_FEDAVG, marker="o", linewidth=2,
            label="FedAvg (no defence)")
    ax.plot(r_lm, [v * 100 for v in a_lm],
            color=COLOR_LMAD, marker="s", linewidth=2,
            label="FedLMAD (proposed)")

    # Annotate final values
    ax.annotate(f"{a_fa[-1]*100:.2f}%",
                xy=(r_fa[-1], a_fa[-1]*100),
                xytext=(8, -14), textcoords="offset points",
                color=COLOR_FEDAVG, fontsize=9, fontweight="bold")
    ax.annotate(f"{a_lm[-1]*100:.2f}%",
                xy=(r_lm[-1], a_lm[-1]*100),
                xytext=(8, 4), textcoords="offset points",
                color=COLOR_LMAD, fontsize=9, fontweight="bold")

    ax.set_xlabel("Federated Round", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title("Centralized Test Accuracy", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=100)

    # ---- Loss panel ----
    ax = axes[1]
    r_fa, l_fa = extract_series(fedavg["centralized_loss"])
    r_lm, l_lm = extract_series(lmad["centralized_loss"])

    ax.plot(r_fa, l_fa,
            color=COLOR_FEDAVG, marker="o", linewidth=2,
            label="FedAvg (no defence)")
    ax.plot(r_lm, l_lm,
            color=COLOR_LMAD, marker="s", linewidth=2,
            label="FedLMAD (proposed)")

    ax.annotate(f"{l_fa[-1]:.4f}",
                xy=(r_fa[-1], l_fa[-1]),
                xytext=(8, 4), textcoords="offset points",
                color=COLOR_FEDAVG, fontsize=9, fontweight="bold")
    ax.annotate(f"{l_lm[-1]:.4f}",
                xy=(r_lm[-1], l_lm[-1]),
                xytext=(8, -14), textcoords="offset points",
                color=COLOR_LMAD, fontsize=9, fontweight="bold")

    ax.set_xlabel("Federated Round", fontsize=11)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax.set_title("Centralized Test Loss", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xlim(left=0)

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "comparison_combined.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved: {out}")
    plt.close(fig)


def plot_single(fedavg: dict, lmad: dict, metric: str) -> None:
    """Single-panel accuracy or loss figure."""
    assert metric in ("accuracy", "loss")
    is_acc = metric == "accuracy"
    key = f"centralized_{metric}"

    fig, ax = plt.subplots(figsize=(7, 5))

    r_fa, v_fa = extract_series(fedavg[key])
    r_lm, v_lm = extract_series(lmad[key])

    if is_acc:
        v_fa = [v * 100 for v in v_fa]
        v_lm = [v * 100 for v in v_lm]

    ax.plot(r_fa, v_fa, color=COLOR_FEDAVG, marker="o", linewidth=2,
            label="FedAvg (no defence)")
    ax.plot(r_lm, v_lm, color=COLOR_LMAD,   marker="s", linewidth=2,
            label="FedLMAD (proposed)")

    ax.set_xlabel("Federated Round", fontsize=12)
    ylabel = "Test Accuracy (%)" if is_acc else "Cross-Entropy Loss"
    ax.set_ylabel(ylabel, fontsize=12)
    title = "Centralized Test Accuracy" if is_acc else "Centralized Test Loss"
    ax.set_title(
        f"{title}\n{_title_info(lmad)}",
        fontsize=11,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xlim(left=0)
    if is_acc:
        ax.set_ylim(bottom=0, top=100)

    fig.tight_layout()
    fname = f"comparison_{metric}.png"
    out = os.path.join(RESULTS_DIR, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved: {out}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Plot 3: L-MAD Rejection Heatmap (per layer, per round)
# ──────────────────────────────────────────────────────────────────────────────
def plot_rejection_heatmap(lmad: dict) -> None:
    """
    Heatmap of per-layer rejection counts across rounds.

    Rows = layers, Columns = rounds.
    Cell value = number of clients rejected for that layer in that round.
    Data is read dynamically from the lmad_results.json.
    """
    lmad_metrics = lmad.get("lmad_metrics", {})

    layer_keys = sorted(
        [k for k in lmad_metrics
         if k.startswith("lmad_layer_") and k.endswith("_rejections")],
        key=lambda k: int(k.split("_")[2]),
    )

    if not layer_keys:
        print("[plot] No per-layer rejection data found in lmad_results.json — skipping heatmap.")
        return

    K   = int(lmad["metadata"].get("num_clients", 10))
    tau = lmad["metadata"].get("tau", 3.0)

    # Collect sorted round numbers from the first layer (all layers share the same rounds)
    rounds_sorted = sorted({d["round"] for d in lmad_metrics[layer_keys[0]]})
    round_idx     = {r: i for i, r in enumerate(rounds_sorted)}
    num_layers    = len(layer_keys)
    num_rounds    = len(rounds_sorted)

    # Build the rejection matrix  (layers × rounds)
    rejections = np.zeros((num_layers, num_rounds), dtype=float)
    for i, key in enumerate(layer_keys):
        for entry in lmad_metrics[key]:
            rejections[i, round_idx[entry["round"]]] = entry["value"]

    layer_names = [f"Layer {int(k.split('_')[2])}" for k in layer_keys]

    fig_w = max(8, num_rounds + 2)
    fig_h = max(4, num_layers * 0.75)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(
        rejections,
        cmap="RdYlGn_r",
        vmin=0, vmax=K,
        aspect="auto",
    )

    ax.set_xticks(range(num_rounds))
    ax.set_xticklabels([f"R{r}" for r in rounds_sorted], fontsize=9)
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels(layer_names, fontsize=9)
    ax.set_xlabel("Round", fontsize=11)
    ax.set_title(
        f"L-MAD: Clients Rejected per Layer per Round  (K={K}, tau={tau})",
        fontsize=10,
    )

    for i in range(num_layers):
        for j in range(num_rounds):
            val = int(rejections[i, j])
            text_color = "white" if val >= K * 0.8 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Clients Rejected", fontsize=10)
    tick_step = max(1, K // 5)
    cbar.set_ticks(range(0, K + 1, tick_step))

    fig.tight_layout()
    fname = os.path.join(RESULTS_DIR, "lmad_rejections.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved: {fname}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────
def print_summary(fedavg: dict, lmad: dict) -> None:
    print("\n" + "=" * 60)
    print(f"{'RESULTS SUMMARY':^60}")
    print("=" * 60)
    print(f"{'Round':<8} {'FedAvg Acc':>12} {'L-MAD Acc':>12} {'Δ Acc':>8}")
    print("-" * 60)

    fa_acc = {d["round"]: d["accuracy"] for d in fedavg["centralized_accuracy"]}
    lm_acc = {d["round"]: d["accuracy"] for d in lmad["centralized_accuracy"]}

    for r in sorted(fa_acc):
        fa = fa_acc[r]
        lm = lm_acc.get(r, float("nan"))
        delta = lm - fa
        marker = " ✓" if delta > 0 else (" ✗" if delta < 0 else "")
        print(f"  {r:<6} {fa*100:>10.2f}%  {lm*100:>10.2f}%  {delta*100:>+7.2f}%{marker}")

    print("-" * 60)
    final_fa = fedavg["centralized_accuracy"][-1]["accuracy"]
    final_lm = lmad["centralized_accuracy"][-1]["accuracy"]
    delta_final = final_lm - final_fa
    print(f"  Final accuracy improvement (L-MAD vs FedAvg): "
          f"{delta_final*100:+.2f} pp")
    print("=" * 60)

    fa_loss = fedavg["centralized_loss"][-1]["loss"]
    lm_loss = lmad["centralized_loss"][-1]["loss"]
    print(f"\n  Final loss — FedAvg: {fa_loss:.4f}  |  L-MAD: {lm_loss:.4f}  "
          f"(Δ {lm_loss - fa_loss:+.4f})")
    print()

    print("  KEY OBSERVATION: most-rejected layer per round")
    lmad_metrics = lmad.get("lmad_metrics", {})
    layer_keys = sorted(
        [k for k in lmad_metrics
         if k.startswith("lmad_layer_") and k.endswith("_rejections")],
        key=lambda k: int(k.split("_")[2]),
    )
    if layer_keys:
        layer_means = {
            k: np.mean([d["value"] for d in lmad_metrics[k]])
            for k in layer_keys
        }
        top_key = max(layer_means, key=layer_means.get)
        top_idx = int(top_key.split("_")[2])
        vals    = [d["value"] for d in lmad_metrics[top_key]]
        print(f"  Layer {top_idx} had the most rejections on average: "
              f"{layer_means[top_key]:.1f}/round "
              f"(min={int(min(vals))}, max={int(max(vals))})")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(FEDAVG_FILE):
        raise FileNotFoundError(f"Missing: {FEDAVG_FILE}")
    if not os.path.exists(LMAD_FILE):
        raise FileNotFoundError(f"Missing: {LMAD_FILE}")

    fedavg = load_json(FEDAVG_FILE)
    lmad   = load_json(LMAD_FILE)

    print_summary(fedavg, lmad)
    plot_combined(fedavg, lmad)
    plot_single(fedavg, lmad, "accuracy")
    plot_single(fedavg, lmad, "loss")
    plot_rejection_heatmap(lmad)

    print("[plot] All figures saved to results/")
