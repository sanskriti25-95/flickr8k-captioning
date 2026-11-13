"""Make the report figures from saved outputs: loss curves, architecture diagram, BLEU chart."""
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
OUT, FIG = ROOT / "outputs", ROOT / "figures"
FIG.mkdir(exist_ok=True)
BLUE, ORANGE, GREY = "#1f4e79", "#d9822b", "#555555"
ENC = {"inception": "InceptionV3", "resnet50": "ResNet50"}


def loss_curves():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, (k, name) in zip(axes, ENC.items()):
        rows = list(csv.DictReader(open(OUT / f"history_{k}.csv")))
        ep = [int(r["epoch"]) + 1 for r in rows]
        tr = [float(r["loss"]) for r in rows]; va = [float(r["val_loss"]) for r in rows]
        ax.plot(ep, tr, "o-", color=BLUE, label="Training loss")
        ax.plot(ep, va, "s-", color=ORANGE, label="Validation loss")
        best = min(range(len(va)), key=va.__getitem__)
        ax.axvline(ep[best], ls=":", color=GREY); ax.annotate(f"best: epoch {ep[best]}\n{va[best]:.3f}", (ep[best], va[best]),
                     xytext=(ep[best] + 0.3, va[best] + 0.45), fontsize=8, color=GREY)
        ax.set_title(name); ax.set_xlabel("Epoch"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Cross-entropy loss"); axes[0].legend()
    fig.tight_layout(); fig.savefig(FIG / "loss_curves.png", dpi=170); plt.close(fig)


def box(ax, x, y, w, h, text, fc="#eef3f8", ec=BLUE, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, color=GREY, lw=1.2))


def architecture():
    fig, ax = plt.subplots(figsize=(9, 4.3)); ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    # image branch (top)
    box(ax, 0.1, 3.6, 1.4, 0.9, "Input image\n299x299 / 224x224", "#ffffff", GREY)
    box(ax, 1.9, 3.6, 1.9, 0.9, "Frozen CNN encoder\nInceptionV3 / ResNet50\n(avg pool)", "#dce8f4")
    box(ax, 4.2, 3.6, 1.3, 0.9, "2048-d\nfeature vector", "#ffffff", GREY)
    box(ax, 5.9, 3.6, 1.5, 0.9, "Dropout 0.5\nDense 256, ReLU")
    # text branch (bottom)
    box(ax, 0.1, 0.5, 1.4, 0.9, "Prefix tokens\n(startseq w1 .. wt)\nlength 34", "#ffffff", GREY)
    box(ax, 1.9, 0.5, 1.5, 0.9, "Embedding 256\n(masked pad)")
    box(ax, 3.8, 0.5, 1.5, 0.9, "Dropout 0.5\nLSTM 256")
    # merge
    box(ax, 7.8, 2.05, 1.0, 0.9, "Add", "#fbe9d6", ORANGE)
    box(ax, 7.8, 0.5, 1.0, 0.9, "Dense 256\nReLU")
    box(ax, 9.0, 0.5, 0.95, 0.9, "Dense 2527\nsoftmax", "#fbe9d6", ORANGE, 8)
    ax.text(9.475, 1.55, "next word", ha="center", fontsize=8.5, color=ORANGE)
    for p, q in [((1.5, 4.05), (1.9, 4.05)), ((3.8, 4.05), (4.2, 4.05)), ((5.5, 4.05), (5.9, 4.05)),
                 ((1.5, 0.95), (1.9, 0.95)), ((3.4, 0.95), (3.8, 0.95)),
                 ((7.4, 4.05), (8.3, 2.95)), ((5.3, 0.95), (7.8, 2.2)),
                 ((8.3, 2.05), (8.3, 1.4)), ((8.8, 0.95), (9.0, 0.95))]:
        arrow(ax, p, q)
    ax.text(5.0, 4.85, "Image branch (features cached once)", ha="center", fontsize=9, color=BLUE, weight="bold")
    ax.text(3.0, 0.1, "Text branch (LSTM language model)", ha="center", fontsize=9, color=BLUE, weight="bold")
    ax.text(5.0, 2.35, "Generation: feed the predicted word back in until 'endseq'\n(greedy or beam search, k = 3)",
            ha="center", fontsize=8, color=GREY, style="italic")
    fig.tight_layout(); fig.savefig(FIG / "architecture.png", dpi=170); plt.close(fig)


def bleu_chart():
    rows = []
    for k, name in ENC.items():
        b = json.load(open(OUT / f"baseline_{k}.json")); e = json.load(open(OUT / f"eval_{k}.json"))
        rows += [(f"{name}\nNN-1", b["nn1"]), (f"{name}\nkNN-5", b["knn5_consensus"]),
                 (f"{name}\ngreedy", e["greedy"]), (f"{name}\nbeam 3", e["beam3"])]
    fig, ax = plt.subplots(figsize=(9, 3.6)); n = len(rows); w = 0.2
    for i, m in enumerate(["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]):
        ax.bar([j + (i - 1.5) * w for j in range(n)], [r[1][m] for r in rows], w, label=m,
               color=["#9bb8d3", "#5b8bb8", "#2f6396", "#12335a"][i])
    ax.set_xticks(range(n)); ax.set_xticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_ylim(0, 0.68); ax.set_ylabel("BLEU (test set)"); ax.legend(ncol=4, fontsize=8, loc="upper left"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "bleu_comparison.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    loss_curves(); architecture(); bleu_chart(); print("figures written to", FIG)
