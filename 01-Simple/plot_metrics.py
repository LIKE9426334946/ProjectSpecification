from pathlib import Path
import csv

import matplotlib.pyplot as plt

# ==========================================
# Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"

METRICS_PATH = RESULT_DIR / "metrics.csv"

LOSS_FIGURE_PATH = RESULT_DIR / "loss_curve.png"
METRICS_FIGURE_PATH = RESULT_DIR / "segmentation_metrics.png"


# ==========================================
# Load CSV
# ==========================================

epochs = []

train_loss = []
val_loss = []

iou = []
f1 = []
precision = []
recall = []


with open(
    METRICS_PATH,
    "r",
    encoding="utf-8",
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        epochs.append(int(row["epoch"]))

        train_loss.append(float(row["train_loss"]))

        val_loss.append(float(row["val_loss"]))

        iou.append(float(row["iou"]))

        f1.append(float(row["f1"]))

        precision.append(float(row["precision"]))

        recall.append(float(row["recall"]))


# ==========================================
# Global Style
# ==========================================

plt.style.use("dark_background")

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 18,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 0.8,
        "lines.linewidth": 2.5,
        "lines.markersize": 5,
    }
)


# ==========================================
# Figure 1
# Loss Curve
# ==========================================

fig, ax = plt.subplots(
    figsize=(11, 6),
    dpi=150,
)

fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")


ax.plot(
    epochs,
    train_loss,
    marker="o",
    label="Train Loss",
)

ax.plot(
    epochs,
    val_loss,
    marker="o",
    label="Validation Loss",
)


# Fill areas

ax.fill_between(
    epochs,
    train_loss,
    alpha=0.08,
)

ax.fill_between(
    epochs,
    val_loss,
    alpha=0.08,
)


# Labels

ax.set_title(
    "Training & Validation Loss",
    pad=20,
    fontweight="bold",
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")


# Grid

ax.grid(
    True,
    linestyle="--",
    linewidth=0.6,
    alpha=0.18,
)


# Legend

legend = ax.legend(
    loc="upper right",
    frameon=True,
)

legend.get_frame().set_alpha(0.15)


# Remove unnecessary borders

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# Best validation point

best_epoch_index = val_loss.index(min(val_loss))

best_epoch = epochs[best_epoch_index]
best_loss = val_loss[best_epoch_index]


ax.scatter(
    best_epoch,
    best_loss,
    s=90,
    zorder=5,
)

ax.annotate(
    f"Best\n{best_loss:.4f}",
    xy=(best_epoch, best_loss),
    xytext=(15, 20),
    textcoords="offset points",
    arrowprops={
        "arrowstyle": "->",
        "alpha": 0.7,
    },
)


fig.tight_layout()

fig.savefig(
    LOSS_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# ==========================================
# Figure 2
# Segmentation Metrics
# ==========================================

fig, ax = plt.subplots(
    figsize=(11, 6),
    dpi=150,
)

fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")


ax.plot(
    epochs,
    iou,
    marker="o",
    label="IoU",
)

ax.plot(
    epochs,
    f1,
    marker="o",
    label="F1 / Dice",
)

ax.plot(
    epochs,
    precision,
    marker="o",
    label="Precision",
)

ax.plot(
    epochs,
    recall,
    marker="o",
    label="Recall",
)


ax.set_title(
    "Segmentation Performance",
    pad=20,
    fontweight="bold",
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Score")


# Metrics range

ax.set_ylim(0, 1.02)


# Grid

ax.grid(
    True,
    linestyle="--",
    linewidth=0.6,
    alpha=0.18,
)


# Legend

legend = ax.legend(
    loc="lower right",
    frameon=True,
    ncol=2,
)

legend.get_frame().set_alpha(0.15)


# Remove unnecessary borders

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# ==========================================
# Highlight best IoU
# ==========================================

best_iou_index = iou.index(max(iou))

best_epoch = epochs[best_iou_index]
best_iou = iou[best_iou_index]


ax.scatter(
    best_epoch,
    best_iou,
    s=100,
    zorder=5,
)

ax.annotate(
    f"Best IoU\n{best_iou:.4f}",
    xy=(best_epoch, best_iou),
    xytext=(15, -45),
    textcoords="offset points",
    arrowprops={
        "arrowstyle": "->",
        "alpha": 0.7,
    },
)


fig.tight_layout()

fig.savefig(
    METRICS_FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.show()
