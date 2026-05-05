#!/usr/bin/env python3
"""
Generate 4 Week 10 P01 lecture figures.

Output PNGs → projects/p01_unet_detector/figures/
  - fig_p01_signal_chain.png    (HTML Fig. 2)
  - fig_p01_unet_architecture.png (HTML Fig. 4)
  - fig_p01_detection_zoom.png  (HTML Fig. 8)
  - fig_p01_fp_zoom.png         (HTML Fig. 9)

Run from repo root:
  python projects/p01_unet_detector/figures/gen_week10_figures.py

Requirements: matplotlib, numpy, h5py (only for fp_zoom)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
P01_ROOT = REPO_ROOT / "projects" / "p01_unet_detector"
OUT_DIR = P01_ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── plot_style ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(REPO_ROOT / "shared"))
from plot_style import BLUE, CYAN, GRAY, GREEN, LIGHT_GRAY, ORANGE, PURPLE, RED, setup_style

setup_style()

# ── Artifact root ──────────────────────────────────────────────────────────
ARTIFACTS = P01_ROOT / "artifacts"
CASE_DIR = ARTIFACTS / "case_studies"


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Signal Chain (fig_p01_signal_chain.png)
# ══════════════════════════════════════════════════════════════════════════════
def fig_signal_chain():
    """Clean 2-row flow diagram: Scene→…→RDM, then 3-way branch."""
    fig, ax = plt.subplots(figsize=(16, 5), constrained_layout=True)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # ── Top-row boxes ──────────────────────────────────────────────────────
    top_stages = [
        ("Scene", "moving targets\n+ static clutter", BLUE),
        ("FMCW beat", "dechirp mixing", BLUE),
        ("ADC I/Q", "complex 16-bit\nfixed scale", CYAN),
        ("MTI", "slow-time\nmean removal", CYAN),
        ("RDM", "range FFT\n× Doppler FFT", ORANGE),
    ]
    n = len(top_stages)
    xs = np.linspace(1.0, 11.5, n)
    bw, bh = 1.6, 0.9
    branch_y = 3.2   # vertical centre of top row
    schema_note_y = 1.6

    box_centres = []
    for i, (label, sub, color) in enumerate(top_stages):
        x = xs[i]
        box_centres.append(x)
        rect = FancyBboxPatch(
            (x - bw / 2, branch_y - bh / 2), bw, bh,
            boxstyle="round,pad=0.08", linewidth=1.2,
            edgecolor=color, facecolor=color + "22",
        )
        ax.add_patch(rect)
        ax.text(x, branch_y + 0.1, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color=color)
        ax.text(x, branch_y - 0.32, sub, ha="center", va="center",
                fontsize=8, color=GRAY, style="italic", linespacing=1.3)

    # arrows between top-row boxes
    for i in range(len(box_centres) - 1):
        x0 = box_centres[i] + bw / 2
        x1 = box_centres[i + 1] - bw / 2
        ax.annotate("", xy=(x1, branch_y), xytext=(x0, branch_y),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY,
                                   lw=1.5, mutation_scale=14))

    # ── Schema-v9 footer note near MTI/RDM ────────────────────────────────
    ax.text(
        8.0,
        0.18,
        "schema-v9: fs/B = 4, static-only clutter, ADC clip = 0",
        ha="center", va="bottom", fontsize=8, color=GRAY,
        style="italic",
    )

    # ── Branch node below RDM ──────────────────────────────────────────────
    rdm_x = box_centres[-1]
    branch_node_y = 2.1
    ax.annotate("", xy=(rdm_x, branch_node_y + 0.18), xytext=(rdm_x, branch_y - bh / 2),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.5, mutation_scale=14))
    circle = plt.Circle((rdm_x, branch_node_y), 0.12,
                         color=ORANGE, zorder=5)
    ax.add_patch(circle)

    # ── Bottom-row branch boxes ────────────────────────────────────────────
    branch_specs = [
        ("Label", "6 dB gate\n5-cell cross", GREEN, rdm_x - 2.6),
        ("CA-CFAR", "local threshold\nlocked policy", PURPLE, rdm_x),
        ("U-Net", "probability map\nthreshold", BLUE, rdm_x + 2.6),
    ]
    bot_y = 0.95
    for label, sub, color, bx in branch_specs:
        # line from branch node
        ax.annotate("", xy=(bx, bot_y + bh / 2),
                    xytext=(rdm_x, branch_node_y - 0.12),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=1.4, mutation_scale=13,
                                   connectionstyle="arc3,rad=0"))
        rect = FancyBboxPatch(
            (bx - bw / 2, bot_y - bh / 2), bw, bh,
            boxstyle="round,pad=0.08", linewidth=1.2,
            edgecolor=color, facecolor=color + "22",
        )
        ax.add_patch(rect)
        ax.text(bx, bot_y + 0.12, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color=color)
        ax.text(bx, bot_y - 0.28, sub, ha="center", va="center",
                fontsize=8, color=GRAY, style="italic", linespacing=1.3)

    out = OUT_DIR / "fig_p01_signal_chain.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — U-Net Architecture (fig_p01_unet_architecture.png)
# ══════════════════════════════════════════════════════════════════════════════
def fig_unet_architecture():
    """U-Net diagram: Enc1-4 → Bottleneck → Dec4-1 → Output, skip connections."""
    fig, ax = plt.subplots(figsize=(18, 6.5), constrained_layout=True)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    # ── Layout constants ───────────────────────────────────────────────────
    bw, bh_base = 1.55, 0.78
    enc_xs   = [1.0, 2.8, 4.6, 6.4]
    bot_x    = 8.2
    dec_xs   = [10.0, 11.9, 13.8, 15.7]   # mirrored, spacing=1.9 > bw=1.55
    centre_y = 3.2

    enc_specs = [
        ("Enc 1", "2→32\n64×200\nConv-BN-ReLU×2", BLUE),
        ("Enc 2", "32→64\n32×100\nConv-BN-ReLU×2", BLUE),
        ("Enc 3", "64→128\n16×50\nConv-BN-ReLU×2", BLUE),
        ("Enc 4", "128→256\n8×25\nConv-BN-ReLU×2", BLUE),
    ]
    dec_specs = [
        ("Dec 4", "256→128\n16×50\nConv-BN-ReLU×2", PURPLE),
        ("Dec 3", "128→64\n32×100\nConv-BN-ReLU×2", PURPLE),
        ("Dec 2", "64→32\n64×200\nConv-BN-ReLU×2", PURPLE),
        ("Dec 1", "32→16\n64×200\nConv-BN-ReLU×2", PURPLE),
    ]

    def draw_box(cx, cy, bh, label, detail, color, bold=True):
        rect = FancyBboxPatch(
            (cx - bw / 2, cy - bh / 2), bw, bh,
            boxstyle="round,pad=0.07", linewidth=1.3,
            edgecolor=color, facecolor=color + "1a",
        )
        ax.add_patch(rect)
        n_lines = detail.count("\n") + 1
        top_offset = bh * 0.26
        ax.text(cx, cy + top_offset, label, ha="center", va="center",
                fontsize=10, fontweight="bold" if bold else "normal", color=color)
        ax.text(cx, cy - 0.05, detail, ha="center", va="center",
                fontsize=7.5, color=GRAY, linespacing=1.35)

    # ── Encoder boxes ─────────────────────────────────────────────────────
    bh_enc = [bh_base + 0.5, bh_base + 0.3, bh_base + 0.1, bh_base]
    enc_positions = []
    for i, (ex, (label, detail, color), bh) in enumerate(
            zip(enc_xs, enc_specs, bh_enc)):
        y = centre_y + 0.15 * i
        draw_box(ex, y, bh, label, detail, color)
        enc_positions.append((ex, y, bh))

    # arrows enc→enc (downsampling arrows go right and slightly down)
    for i in range(len(enc_xs) - 1):
        x0, y0, bh0 = enc_positions[i]
        x1, y1, bh1 = enc_positions[i + 1]
        ax.annotate("", xy=(x1 - bw / 2, y1), xytext=(x0 + bw / 2, y0),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE,
                                   lw=1.3, mutation_scale=12))

    # ── Bottleneck ─────────────────────────────────────────────────────────
    bot_bh = bh_base + 0.1
    bot_y = centre_y + 0.15 * 3 - 0.2
    draw_box(bot_x, bot_y, bot_bh + 0.2,
             "Bottleneck", "256→512\n4×13\nConv-BN-ReLU×2\nDropout2d(0.3)", GREEN, bold=True)
    # arrow enc4→bottleneck
    x0, y0, bh0 = enc_positions[-1]
    ax.annotate("", xy=(bot_x - bw / 2, bot_y), xytext=(x0 + bw / 2, y0),
                arrowprops=dict(arrowstyle="-|>", color=GREEN,
                               lw=1.3, mutation_scale=12))

    # ── Decoder boxes ─────────────────────────────────────────────────────
    dec_positions = []
    for i, (dx, (label, detail, color), bh) in enumerate(
            zip(dec_xs, dec_specs, reversed(bh_enc))):
        y = centre_y + 0.15 * (3 - i)
        draw_box(dx, y, bh, label, detail, color)
        dec_positions.append((dx, y, bh))

    # arrow bottleneck→dec4
    dx0, dy0, dbh0 = dec_positions[0]
    ax.annotate("", xy=(dx0 - bw / 2, dy0), xytext=(bot_x + bw / 2, bot_y),
                arrowprops=dict(arrowstyle="-|>", color=GREEN,
                               lw=1.3, mutation_scale=12))
    # arrows dec→dec
    for i in range(len(dec_positions) - 1):
        x0, y0, bh0 = dec_positions[i]
        x1, y1, bh1 = dec_positions[i + 1]
        ax.annotate("", xy=(x1 - bw / 2, y1), xytext=(x0 + bw / 2, y0),
                    arrowprops=dict(arrowstyle="-|>", color=PURPLE,
                                   lw=1.3, mutation_scale=12))

    # ── Output box ────────────────────────────────────────────────────────
    out_x = dec_xs[-1] + 1.6
    out_y = dec_positions[-1][1]
    draw_box(out_x, out_y, bh_base - 0.1,
             "Output", "1×64×200\nSigmoid", RED, bold=True)
    ax.annotate("", xy=(out_x - bw / 2, out_y),
                xytext=(dec_positions[-1][0] + bw / 2, dec_positions[-1][1]),
                arrowprops=dict(arrowstyle="-|>", color=RED,
                               lw=1.5, mutation_scale=13))

    # ── Skip connections ──────────────────────────────────────────────────
    skip_pairs = list(zip(enc_positions, dec_positions[::-1]))
    skip_colors = [BLUE, BLUE, BLUE, BLUE]
    for (ex, ey, ebh), (dx, dy, dbh) in skip_pairs:
        # draw dashed arc above
        skip_y_top = max(ey + ebh / 2, dy + dbh / 2) + 0.45
        ax.annotate("", xy=(dx, dy + dbh / 2 + 0.08),
                    xytext=(ex, ey + ebh / 2 + 0.08),
                    arrowprops=dict(
                        arrowstyle="-|>", color=ORANGE,
                        lw=1.1, mutation_scale=11,
                        linestyle="dashed",
                        connectionstyle=f"arc3,rad=-0.35",
                    ))
        # concat label near decoder
        ax.text(dx - bw * 0.38, dy + dbh / 2 + 0.18,
                "cat", fontsize=7, color=ORANGE, ha="center")

    # ── Skip connection legend ────────────────────────────────────────────
    # Footer area (bottom left)
    footer_y = 0.65
    footer_x = 0.4
    ax.text(footer_x, footer_y + 0.35, "Legend", fontsize=9,
            fontweight="bold", color=GRAY, va="bottom")
    legend_items = [
        (BLUE, "Encoder block (Conv-BN-ReLU ×2)"),
        (PURPLE, "Decoder block (Conv-BN-ReLU ×2, reflect-pad→crop)"),
        (ORANGE, "Skip connection (concat)"),
        (GREEN, "Bottleneck (Dropout2d)"),
        (RED, "Output (Sigmoid)"),
    ]
    for j, (col, txt) in enumerate(legend_items):
        rect = mpatches.Patch(facecolor=col + "33", edgecolor=col, linewidth=1)
        ax.text(footer_x + 0.35, footer_y - j * 0.28, "■", color=col, fontsize=10, va="center")
        ax.text(footer_x + 0.62, footer_y - j * 0.28, txt, fontsize=8, color=GRAY, va="center")

    # params note bottom right
    ax.text(17.8, 0.25, "≈ 7.76 M parameters", fontsize=8.5,
            color=GRAY, ha="right", style="italic")

    out = OUT_DIR / "fig_p01_unet_architecture.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Detection Zoom (fig_p01_detection_zoom.png)
# ══════════════════════════════════════════════════════════════════════════════
def fig_detection_zoom():
    """2×3 close-up grid: strong target (scene_01 t0) and weak target (scene_01 t1)."""
    sc = np.load(CASE_DIR / "scene_01.npz", allow_pickle=True)

    rdm = sc["rdm_log_mag"].astype(float)   # (64, 200)
    gt  = sc["gt_mask"].astype(bool)
    cfar = sc["cfar_det"].astype(bool)
    unet = sc["unet_det"].astype(bool)
    vel_axis   = sc["velocity_axis_mps"]
    range_axis = sc["range_axis_m"]
    n_targets  = int(sc["n_targets"])
    t_range    = sc["target_range"][:n_targets]
    t_vel      = sc["target_velocity"][:n_targets]

    # find GT peak bins for each target
    def find_gt_bin(trange, tvel):
        ri = int(np.argmin(np.abs(range_axis - trange)))
        vi = int(np.argmin(np.abs(vel_axis - tvel)))
        return vi, ri   # (doppler_idx, range_idx) = (row, col)

    t0_vi, t0_ri = find_gt_bin(t_range[0], t_vel[0])   # strong target
    t1_vi, t1_ri = find_gt_bin(t_range[1], t_vel[1])   # weak target

    # crop half-widths in bins
    dv = int(12 / (vel_axis[1] - vel_axis[0]))    # ±12 m/s → bins
    dr = int(20 / (range_axis[1] - range_axis[0])) # ±20 m → bins

    def crop(arr, vi, ri, dv=dv, dr=dr):
        r0, r1 = max(0, ri - dr), min(arr.shape[1], ri + dr + 1)
        v0, v1 = max(0, vi - dv), min(arr.shape[0], vi + dv + 1)
        return arr[v0:v1, r0:r1], vel_axis[v0:v1], range_axis[r0:r1]

    rdm0, vel0, rng0 = crop(rdm,  t0_vi, t0_ri)
    gt0, _, _        = crop(gt,   t0_vi, t0_ri)
    cfar0, _, _      = crop(cfar, t0_vi, t0_ri)
    unet0, _, _      = crop(unet, t0_vi, t0_ri)

    rdm1, vel1, rng1 = crop(rdm,  t1_vi, t1_ri)
    gt1, _, _        = crop(gt,   t1_vi, t1_ri)
    cfar1, _, _      = crop(cfar, t1_vi, t1_ri)
    unet1, _, _      = crop(unet, t1_vi, t1_ri)

    def local_clim(rdm_crop, gt_crop, margin_db=6):
        """Set clim based on local peak and noise floor."""
        peak = rdm_crop[gt_crop].max() if gt_crop.any() else rdm_crop.max()
        noise = np.percentile(rdm_crop, 20)
        return noise - margin_db * 0.05, peak + margin_db * 0.05

    vmin0, vmax0 = local_clim(rdm0, gt0)
    vmin1, vmax1 = local_clim(rdm1, gt1)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    col_titles = ["Strong target\n(RDM + GT)", "Strong target\n(CA-CFAR vs GT)",
                  "Strong target\n(U-Net vs GT)",
                  "Weak target\n(RDM + GT)", "Weak target\n(CA-CFAR vs GT)",
                  "Weak target\n(U-Net vs GT)"]

    def plot_crop(ax, rdm_crop, rng, vel, gt_crop, det_crop, det_label,
                  vmin, vmax, det_color):
        ext = [rng[0], rng[-1], vel[0], vel[-1]]
        im = ax.imshow(rdm_crop, aspect="auto", origin="lower",
                       extent=ext, cmap="inferno",
                       vmin=vmin, vmax=vmax, interpolation="bilinear")
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="log-mag")
        # GT contour
        gt_ys, gt_xs = np.where(gt_crop)
        if gt_ys.size:
            ax.scatter(rng[gt_xs], vel[gt_ys],
                       marker="o", s=80, facecolors="none",
                       edgecolors="white", linewidths=1.5,
                       label="GT", zorder=5)
        # detections
        det_ys, det_xs = np.where(det_crop)
        if det_ys.size:
            ax.scatter(rng[det_xs], vel[det_ys],
                       marker="x", s=60, c=det_color,
                       linewidths=1.4, alpha=0.85,
                       label=det_label, zorder=6)
        ax.set_xlabel("Range (m)", fontsize=9)
        ax.set_ylabel("Velocity (m/s)", fontsize=9)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.6)

    # Row 0: strong target
    axes[0, 0].set_title("Strong target — RDM + GT", fontsize=10)
    plot_crop(axes[0, 0], rdm0, rng0, vel0, gt0, gt0, "GT", vmin0, vmax0, GREEN)
    axes[0, 1].set_title("Strong target — CA-CFAR", fontsize=10)
    plot_crop(axes[0, 1], rdm0, rng0, vel0, gt0, cfar0, "CA-CFAR", vmin0, vmax0, RED)
    axes[0, 2].set_title("Strong target — U-Net", fontsize=10)
    plot_crop(axes[0, 2], rdm0, rng0, vel0, gt0, unet0, "U-Net", vmin0, vmax0, BLUE)

    # Row 1: weak target (separate clim)
    axes[1, 0].set_title("Weak target — RDM + GT", fontsize=10)
    plot_crop(axes[1, 0], rdm1, rng1, vel1, gt1, gt1, "GT", vmin1, vmax1, GREEN)
    axes[1, 1].set_title("Weak target — CA-CFAR", fontsize=10)
    plot_crop(axes[1, 1], rdm1, rng1, vel1, gt1, cfar1, "CA-CFAR", vmin1, vmax1, RED)
    axes[1, 2].set_title("Weak target — U-Net", fontsize=10)
    plot_crop(axes[1, 2], rdm1, rng1, vel1, gt1, unet1, "U-Net", vmin1, vmax1, BLUE)

    out = OUT_DIR / "fig_p01_detection_zoom.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — FP Pattern (fig_p01_fp_zoom.png)
# ══════════════════════════════════════════════════════════════════════════════
def fig_fp_zoom():
    """
    Multi-scene FP pattern comparison.
    Uses all 4 case_study scenes (scene_01, scene_02 have real CFAR FPs).
    For each scene: left = CA-CFAR, right = U-Net.
    TP=green circle, FP=red x, GT center=white +
    """
    scene_ids = [0, 1, 2, 3]
    scenes = []
    for sid in scene_ids:
        sc = np.load(CASE_DIR / f"scene_{sid:02d}.npz", allow_pickle=True)
        scenes.append(sc)

    n_scenes = len(scenes)
    fig, axes = plt.subplots(n_scenes, 2, figsize=(11, 3.2 * n_scenes),
                             constrained_layout=True)

    snr_labels = [f"SNR {float(sc['snr_db']):.0f} dB" for sc in scenes]

    for row, sc in enumerate(scenes):
        rdm = sc["rdm_log_mag"].astype(float)
        gt  = sc["gt_mask"].astype(bool)
        cfar = sc["cfar_det"].astype(bool)
        unet = sc["unet_det"].astype(bool)
        vel_axis   = sc["velocity_axis_mps"]
        range_axis = sc["range_axis_m"]
        n_targets  = int(sc["n_targets"])
        t_range    = sc["target_range"][:n_targets]
        t_vel      = sc["target_velocity"][:n_targets]

        ext = [range_axis[0], range_axis[-1], vel_axis[0], vel_axis[-1]]

        def compute_tp_fp(det, gt_mask):
            tp_y, tp_x = np.where(det & gt_mask)
            fp_y, fp_x = np.where(det & ~gt_mask)
            return tp_y, tp_x, fp_y, fp_x

        for col, (det, det_name) in enumerate([(cfar, "CA-CFAR"), (unet, "U-Net")]):
            ax = axes[row, col]
            tp_y, tp_x, fp_y, fp_x = compute_tp_fp(det, gt)

            # RDM background
            vmin = np.percentile(rdm, 5)
            vmax = np.percentile(rdm, 99)
            ax.imshow(rdm, aspect="auto", origin="lower", extent=ext,
                      cmap="gray", vmin=vmin, vmax=vmax,
                      interpolation="bilinear", alpha=0.85)

            # GT centers (white +)
            for tr, tv in zip(t_range, t_vel):
                ax.plot(tr, tv, "+", color="white", ms=12, mew=2, zorder=8)

            # TP (green circle)
            if tp_y.size:
                ax.scatter(range_axis[tp_x], vel_axis[tp_y],
                           marker="o", s=55, facecolors="none",
                           edgecolors=GREEN, linewidths=1.5, zorder=7,
                           label="TP")
            # FP (red x)
            if fp_y.size:
                ax.scatter(range_axis[fp_x], vel_axis[fp_y],
                           marker="x", s=55, c=RED,
                           linewidths=1.5, zorder=7, label="FP")

            ax.set_title(f"Scene {row} ({snr_labels[row]}) — {det_name}", fontsize=10)
            ax.set_xlabel("Range (m)", fontsize=9)
            ax.set_ylabel("Velocity (m/s)", fontsize=9)

            # TP/FP count annotation
            ax.text(0.02, 0.97,
                    f"TP {len(tp_y)}  FP {len(fp_y)}",
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=8.5, color="white",
                    bbox=dict(facecolor="black", alpha=0.45, pad=2, edgecolor="none"))

            if tp_y.size or fp_y.size:
                ax.legend(fontsize=8, loc="upper right",
                          framealpha=0.55, labelcolor="white",
                          labelspacing=0.3,
                          facecolor="#111111")

    out = OUT_DIR / "fig_p01_fp_zoom.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating Week 10 P01 figures...")
    fig_signal_chain()
    fig_unet_architecture()
    fig_detection_zoom()
    fig_fp_zoom()
    print("\nAll done. Output directory:", OUT_DIR)
