"""Plot per-round training loss curves from train_loss_curves_<timestamp>.json.

Usage:
    python src/models/plot_train_loss_curves.py
    python src/models/plot_train_loss_curves.py artifacts/global/train_loss_curves_2026-05-20T14-30-00.json
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def find_latest_curves_file(artifacts_dir: Path) -> Path:
    files = sorted(artifacts_dir.glob("train_loss_curves_*.json"))
    if not files:
        raise FileNotFoundError(f"No train_loss_curves_*.json found in {artifacts_dir}")
    return files[-1]


def plot_curves(curves_path: Path, out_path: Path | None = None) -> None:
    with open(curves_path) as f:
        data = json.load(f)

    curves: dict[str, dict] = data["curves"]
    train_start = data.get("train_start_date", "?")
    train_end = data.get("train_end_date", "?")
    training_rows = data.get("training_rows", 0)

    quantile_keys = sorted(curves.keys())
    n = len(quantile_keys)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()

    for i, q_key in enumerate(quantile_keys):
        ax = axes[i]
        entry = curves[q_key]
        q_label = f"q={int(q_key[1:]) / 100:.2f}"

        train_vals = entry["train"]
        best_round = entry.get("best_round", len(train_vals))
        rounds = list(range(len(train_vals)))

        ax.plot(rounds, train_vals, linewidth=1.5, color="#2563eb", label="train loss (1% sample)")
        ax.axvline(best_round - 1, color="#dc2626", linestyle="--", linewidth=1, alpha=0.8,
                   label=f"elbow × 1.2 → round {best_round}")

        total_drop = train_vals[0] - train_vals[-1]
        ax.set_title(
            f"{q_label}\n"
            f"r0={train_vals[0]:.4f} → r{len(train_vals)-1}={train_vals[-1]:.4f}  (Δ={total_drop:.4f})\n"
            f"elbow × 1.2 → best_round={best_round}",
            fontsize=9,
        )

        ax.set_xlabel("Boosting round", fontsize=8)
        ax.set_ylabel("Pinball loss", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"XGBoost training loss curves  |  train {train_start} → {train_end}"
        f"  |  {training_rows:,} rows",
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()

    if out_path is None:
        out_path = curves_path.parent / curves_path.name.replace(".json", ".png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = find_latest_curves_file(Path("artifacts/global"))

    plot_curves(path)
