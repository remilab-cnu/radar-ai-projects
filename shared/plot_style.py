"""Shared matplotlib style for radar AI course figures.

Based on IEEE paper-figure-workflow preset, adapted for lecture web display.
All gen_*.py scripts should call setup_style() before creating figures.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ── Color palette (Tailwind CSS 500-weight, matches website --primary-500) ──
BLUE = '#3b82f6'
RED = '#ef4444'
GREEN = '#10b981'
ORANGE = '#f59e0b'
PURPLE = '#8b5cf6'
GRAY = '#6b7280'
LIGHT_GRAY = '#d1d5db'
CYAN = '#06b6d4'

# Ordered cycle for multi-series plots (colorblind-distinguishable)
COLOR_CYCLE = [BLUE, RED, GREEN, ORANGE, PURPLE, CYAN, GRAY]


def setup_style():
    """Apply IEEE-inspired lecture style globally.

    Differences from strict IEEE journal preset:
    - Slightly larger fonts (11pt base vs 8pt) for screen readability
    - Grid enabled by default for lecture clarity
    - 160 DPI PNG output (web, not 600 DPI journal)
    - Wider default figures (lecture slides, not 3.5" column)
    """
    mpl.rcParams.update({
        # ── Font ──
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'Noto Sans CJK JP',
                            'DejaVu Sans', 'Liberation Sans'],
        'axes.unicode_minus': False,
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,

        # ── Spines (IEEE: no top/right) ──
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,

        # ── Lines & markers ──
        'lines.linewidth': 1.8,
        'lines.markersize': 5,

        # ── Grid ──
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.linewidth': 0.6,

        # ── Figure ──
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'figure.dpi': 100,
        'savefig.dpi': 160,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.04,

        # ── Font embedding (PDF) ──
        'pdf.fonttype': 42,
        'ps.fonttype': 42,

        # ── Color cycle ──
        'axes.prop_cycle': mpl.cycler(color=COLOR_CYCLE),
    })


def add_panel_labels(axes, labels=None, x=-0.02, y=1.05):
    """Add (a), (b), (c), ... panel labels to multi-panel figures.

    Parameters
    ----------
    axes : array-like of Axes
    labels : list of str, optional. Defaults to (a), (b), ...
    x, y : float — position in axes coordinates
    """
    if not isinstance(axes, (list, tuple)):
        try:
            axes = list(axes.flat)
        except AttributeError:
            axes = [axes]
    if labels is None:
        labels = [f'({chr(ord("a") + i)})' for i in range(len(axes))]
    for ax, label in zip(axes, labels):
        ax.text(x, y, label, transform=ax.transAxes,
                va='bottom', ha='left', fontweight='bold', fontsize=12)
