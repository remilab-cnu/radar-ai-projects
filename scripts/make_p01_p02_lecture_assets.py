#!/usr/bin/env python3
"""Build standalone technical figures for P01/P02 lecture preparation.

The figures are intentionally self-contained PNGs.  They do not modify the
course HTML; they only place reusable assets under ``docs/lecture-assets``.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "lecture-assets"
P01_ART = ROOT / "projects" / "p01_unet_detector" / "artifacts" / "full_eval"
P02_ART = ROOT / "projects" / "p02_resnet18_har" / "artifacts" / "stress_eval"


BLUE = "#2563eb"
SKY = "#dbeafe"
TEAL = "#0f766e"
TEAL_LIGHT = "#ccfbf1"
ORANGE = "#f97316"
ORANGE_LIGHT = "#ffedd5"
PURPLE = "#7c3aed"
PURPLE_LIGHT = "#ede9fe"
GREEN = "#16a34a"
GREEN_LIGHT = "#dcfce7"
RED = "#dc2626"
RED_LIGHT = "#fee2e2"
GRAY = "#475569"
GRAY_LIGHT = "#f1f5f9"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def metric(path: Path, key: str, default: float) -> float:
    try:
        return float(load_json(path)["selected_policy"][key])
    except Exception:
        return default


def draw_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fc: str,
    ec: str,
    fontsize: int = 11,
    weight: str = "normal",
) -> None:
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.8,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#0f172a",
        fontweight=weight,
        linespacing=1.25,
    )


def arrow(ax, start, end, *, color: str = GRAY, lw: float = 2.0) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, shrinkA=2, shrinkB=2),
    )


def fig_p01_signal_chain() -> Path:
    cfar_pd = metric(P01_ART / "p01_cfar_selected_test.json", "Pd", 0.5429)
    cfar_f1 = metric(P01_ART / "p01_cfar_selected_test.json", "F1", 0.6095)
    cfar_fa = metric(P01_ART / "p01_cfar_selected_test.json", "false_alarms_per_rdm", 2.466)
    unet_pd = metric(P01_ART / "p01_unet_selected_test.json", "Pd", 0.7740)
    unet_f1 = metric(P01_ART / "p01_unet_selected_test.json", "F1", 0.8322)
    unet_fa = metric(P01_ART / "p01_unet_selected_test.json", "false_alarms_per_rdm", 0.892)

    fig, ax = plt.subplots(figsize=(16, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02,
        0.95,
        "P01 signal chain: controlled FMCW RDM detection after MTI",
        fontsize=20,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        0.02,
        0.905,
        "Static clutter is present in the raw scene, but the supervised task is moving-target detection on the MTI-filtered RDM.",
        fontsize=12.5,
        color=GRAY,
    )

    y = 0.58
    h = 0.15
    w = 0.125
    xs = [0.035, 0.195, 0.355, 0.515, 0.675]
    boxes = [
        ("Scene\nmoving targets\n+ static clutter", GREEN_LIGHT, GREEN),
        ("FMCW beat\nshared dechirp\nsimulator", SKY, BLUE),
        ("Receiver ADC\ncomplex 16-bit I/Q\nfixed full scale", ORANGE_LIGHT, ORANGE),
        ("MTI DC notch\nslow-time mean\nremoval", PURPLE_LIGHT, PURPLE),
        ("Range-Doppler map\nrange FFT\n+ Doppler FFT", GRAY_LIGHT, GRAY),
    ]
    for x, (label, fc, ec) in zip(xs, boxes):
        draw_box(ax, x, y, w, h, label, fc=fc, ec=ec, fontsize=10.5, weight="bold")
    for x0, x1 in zip(xs[:-1], xs[1:]):
        arrow(ax, (x0 + w + 0.006, y + h / 2), (x1 - 0.006, y + h / 2))

    branch_x = xs[-1] + w
    branches = [
        (0.79, 0.76, 0.18, 0.125, "Label gate\npeak / max(global median,\nlocal ring) >= 6 dB", RED_LIGHT, RED),
        (0.79, 0.55, 0.18, 0.125, f"CA-CFAR baseline\nguard=(1,1), train=(4,4)\nPd={cfar_pd:.3f}, F1={cfar_f1:.3f}, FA/RDM={cfar_fa:.3f}", ORANGE_LIGHT, ORANGE),
        (0.79, 0.34, 0.18, 0.125, f"U-Net detector\n2-channel input\nPd={unet_pd:.3f}, F1={unet_f1:.3f}, FA/RDM={unet_fa:.3f}", TEAL_LIGHT, TEAL),
    ]
    for bx, by, bw, bh, label, fc, ec in branches:
        draw_box(ax, bx, by, bw, bh, label, fc=fc, ec=ec, fontsize=9.4, weight="bold")
        arrow(ax, (branch_x + 0.01, y + h / 2), (bx - 0.01, by + bh / 2), color=ec)

    ax.text(
        0.035,
        0.21,
        "Schema-v9 teaching defaults",
        fontsize=14,
        fontweight="bold",
        color="#0f172a",
    )
    bullets = [
        "50k / 5k / 5k split, static clutter only, no multipath ghosts, no Doppler tail",
        "Sampling rate = 4 × signal bandwidth; up/down conversion is outside the teaching simulator",
        "Fixed-scale complex I/Q quantization is checked before MTI/RDM; current clipping fraction is zero",
        "Validation chooses detector settings; test split reports the locked policy",
    ]
    for i, text in enumerate(bullets):
        ax.text(0.052, 0.165 - i * 0.04, f"• {text}", fontsize=11.5, color=GRAY)

    out = OUT / "p01_signal_chain.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_p01_label_mask() -> Path:
    n = 15
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    peak = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / 4.2)
    ridge = 0.18 * np.exp(-((yy - c) ** 2) / 18.0)
    noise = 0.035 * np.sin(xx * 0.9) * np.cos(yy * 0.7)
    patch = 0.18 + ridge + peak + noise
    patch = (patch - patch.min()) / (patch.max() - patch.min())

    mask = np.zeros((n, n), dtype=int)
    for dy, dx in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
        mask[c + dy, c + dx] = 1

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8))
    for ax in axes:
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.1)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    ax = axes[0]
    ax.imshow(patch, cmap="viridis", origin="lower")
    ax.set_title("Processed RDM patch\npeak and local background", pad=12, fontsize=14.5)
    outer = patches.Rectangle((c - 5.5, c - 5.5), 11, 11, fill=False, ec=ORANGE, lw=2.5)
    guard = patches.Rectangle((c - 1.5, c - 1.5), 3, 3, fill=False, ec=RED, lw=2.5, linestyle="--")
    ax.add_patch(outer)
    ax.add_patch(guard)
    ax.scatter([c], [c], s=120, marker="x", color="white", linewidths=3)
    ax.text(0.03, 0.97, "orange: local background ring\nred dashed: guard region", transform=ax.transAxes, va="top", ha="left", fontsize=10, color="white", bbox=dict(boxstyle="round,pad=0.35", fc="black", alpha=0.55))

    ax = axes[1]
    cmap = ListedColormap(["#f8fafc", RED])
    ax.imshow(mask, cmap=cmap, vmin=0, vmax=1, origin="lower")
    ax.set_title("Binary label\nfive-cell cross", pad=12, fontsize=14.5)
    for dy, dx in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
        ax.add_patch(patches.Rectangle((c + dx - 0.5, c + dy - 0.5), 1, 1, fill=False, ec="#7f1d1d", lw=2.5))
    ax.text(
        0.03,
        0.97,
        "positive cells = center + 4-neighbors\nlabel gate >= 6 dB after MTI",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10.5,
        color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cbd5e1", alpha=0.95),
    )

    fig.suptitle("P01 label definition controls what Pd and F1 mean", fontsize=18, fontweight="bold", y=1.01)
    fig.text(
        0.5,
        0.02,
        "A target is labelled only when the processed peak clears max(global median, local ring median) by 6 dB.",
        ha="center",
        fontsize=11.5,
        color=GRAY,
    )
    fig.subplots_adjust(wspace=0.16, top=0.80, bottom=0.12)
    out = OUT / "p01_label_mask_definition.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_p02_target_range_equations() -> Path:
    fig, ax = plt.subplots(figsize=(15, 8.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.94, "P02 target-range micro-Doppler signal model", fontsize=21, fontweight="bold", color="#0f172a")
    ax.text(
        0.03,
        0.895,
        "This path uses a P02-only pedestrian scatterer model and a local range-compressed frame, not a full raw FMCW cube.",
        fontsize=12.5,
        color=GRAY,
    )

    left_x = 0.04
    box_w = 0.27
    box_h = 0.11
    ys = [0.73, 0.57, 0.41, 0.25]
    labels = [
        ("Body kinematics\nBoulic-style activity motion", GREEN_LIGHT, GREEN),
        ("P02 scatterers\nlimbs, torso, head\naspect projection", SKY, BLUE),
        ("Local range-compressed frame\nrange response around target", ORANGE_LIGHT, ORANGE),
        ("Target-bin slow time\nSTFT -> spectrogram", PURPLE_LIGHT, PURPLE),
    ]
    for y, (label, fc, ec) in zip(ys, labels):
        draw_box(ax, left_x, y, box_w, box_h, label, fc=fc, ec=ec, fontsize=11, weight="bold")
    for y0, y1 in zip(ys[:-1], ys[1:]):
        arrow(ax, (left_x + box_w / 2, y0), (left_x + box_w / 2, y1 + box_h), color=GRAY)

    eq_x = 0.40
    ax.text(eq_x, 0.80, "Range and aspect projection", fontsize=14, fontweight="bold", color="#0f172a")
    ax.text(eq_x, 0.745, r"$R_k(t,\theta)=R_0 + x_k(t)\cos\theta + \delta_k(t)\cos\theta$", fontsize=18, color="#0f172a")
    ax.text(eq_x, 0.705, r"Signed aspect is redundant here because the current 2-D model uses $\cos\theta$.", fontsize=11.5, color=GRAY)

    ax.text(eq_x, 0.62, "Local range-compressed response", fontsize=14, fontweight="bold", color="#0f172a")
    ax.text(
        eq_x,
        0.565,
        r"$s(t,r_n)=\sum_k a_k\,\mathrm{sinc}\!\left(\frac{r_n-R_k(t,\theta)}{\Delta R}\right)$",
        fontsize=15.5,
        color="#0f172a",
    )
    ax.text(
        eq_x,
        0.505,
        r"$\qquad\qquad\cdot\,\exp\!\left(-j\,\frac{4\pi R_k(t,\theta)}{\lambda}\right)$",
        fontsize=15.5,
        color="#0f172a",
    )

    ax.text(eq_x, 0.38, "Target-range extraction and spectrogram", fontsize=14, fontweight="bold", color="#0f172a")
    ax.text(eq_x, 0.325, r"$z(t)=s(t,r_{n_\mathrm{target}})$", fontsize=18, color="#0f172a")
    ax.text(eq_x, 0.265, r"$X(\tau,f_D)=|\mathrm{STFT}\{z(t)\}|^2$", fontsize=18, color="#0f172a")

    ax.text(eq_x, 0.165, "Doppler alias guard", fontsize=14, fontweight="bold", color="#0f172a")
    ax.text(eq_x, 0.108, r"$v_\mathrm{max}=\lambda\,\mathrm{PRF}/4$", fontsize=18, color="#0f172a")
    ax.text(eq_x + 0.30, 0.108, "Every generated sample stores its alias margin.", fontsize=11.5, color=GRAY, va="center")

    out = OUT / "p02_target_range_equations.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def _stress_data() -> tuple[list[str], dict[str, list[float]]]:
    path = P02_ART / "p02_stress_comparison_summary_compact.json"
    summary = load_json(path)
    sets = ["Default\n[0,60 deg]", "Aspect\n[60,80 deg]", "Low SNR\n[0,8 dB]", "Far range\n[18,26 m]"]
    stress_keys = [
        "data_stress_aspect_60_80",
        "data_stress_low_snr_0_8",
        "data_stress_far_range_18_26",
    ]
    methods = {
        "RBF SVM": "feature_rbf_svm_10k",
        "TinyCNN": "tiny_cnn",
        "ResNet18": "resnet18",
    }
    values: dict[str, list[float]] = {}
    for label, key in methods.items():
        first = summary["stress_sets"][stress_keys[0]]["methods"][key]["default_accuracy"]
        vals = [100.0 * first]
        for sk in stress_keys:
            vals.append(100.0 * summary["stress_sets"][sk]["methods"][key]["stress_accuracy"])
        values[label] = vals
    return sets, values


def fig_p02_stress_interpretation() -> Path:
    sets, values = _stress_data()

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6), gridspec_kw={"width_ratios": [1.45, 1.0]})
    ax = axes[0]
    x = np.arange(len(sets))
    width = 0.23
    colors = {"RBF SVM": ORANGE, "TinyCNN": TEAL, "ResNet18": BLUE}
    offsets = [-width, 0, width]
    for (label, vals), off in zip(values.items(), offsets):
        bars = ax.bar(x + off, vals, width=width, label=label, color=colors[label], alpha=0.88)
        for b, v in zip(bars, vals):
            if v < 90 or b.get_x() + b.get_width() / 2 < 0.5:
                ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(sets)
    ax.set_ylim(45, 104)
    ax.set_ylabel("Accuracy [%]")
    ax.set_title("P02 default accuracy is saturated; stress sets reveal the lesson", pad=12)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower left", framealpha=0.94)

    ax = axes[1]
    ax.axis("off")
    ax.text(0.00, 0.94, "Lecture interpretation", fontsize=17, fontweight="bold", color="#0f172a")
    points = [
        ("Default IID", "Easy, balanced distribution; many methods look excellent."),
        ("Aspect stress", "Radial Doppler shrinks as cos(theta); handcrafted features drop sharply."),
        ("Low SNR", "Noise hurts handcrafted descriptors more than the trained CNNs in this setup."),
        ("Far range", "Small effect because the generator samples labelled SNR explicitly."),
    ]
    y = 0.82
    for title, body in points:
        ax.text(0.02, y, title, fontsize=13.5, fontweight="bold", color="#0f172a")
        ax.text(0.02, y - 0.055, body, fontsize=11.3, color=GRAY, wrap=True)
        y -= 0.18
    ax.text(
        0.02,
        0.05,
        "Safe claim: P02 teaches controlled target-range micro-Doppler and distribution shift, not a general pedestrian radar benchmark.",
        fontsize=11.2,
        color="#7c2d12",
        bbox=dict(boxstyle="round,pad=0.45", fc="#ffedd5", ec=ORANGE),
        wrap=True,
    )

    out = OUT / "p02_stress_interpretation.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    setup()
    paths = [
        fig_p01_signal_chain(),
        fig_p01_label_mask(),
        fig_p02_target_range_equations(),
        fig_p02_stress_interpretation(),
    ]
    print("Generated lecture assets:")
    for path in paths:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
