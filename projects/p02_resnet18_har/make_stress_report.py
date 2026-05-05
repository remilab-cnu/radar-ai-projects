#!/usr/bin/env python3
"""Create visual P02 stress-generalization report assets.

The script consumes `aggregate_stress_results.py` outputs plus generated stress
HDF5 files and writes PNG figures + a Markdown report suitable for sharing.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.micro_doppler import ACTIVITY_LABELS

METHOD_ORDER = ["feature_logreg", "feature_rbf_svm_10k", "tiny_cnn", "resnet18"]
METHOD_LABELS = {
    "feature_logreg": "LogReg",
    "feature_rbf_svm_10k": "RBF SVM",
    "tiny_cnn": "TinyCNN",
    "resnet18": "ResNet18",
}
CONFUSION_FILES = {
    "feature_logreg": "feature_logreg.json",
    "feature_rbf_svm_10k": "feature_rbf_svm_10k.json",
    "tiny_cnn": "tiny_cnn.json",
    "resnet18": "resnet18.json",
}
DISPLAY_CLASSES = ["walk", "run", "wave", "idle"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def ensure_out(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_accuracy_bars(compact: dict[str, Any], out: Path) -> None:
    stress_names = list(compact["stress_sets"].keys())
    x = np.arange(len(stress_names))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5.6), constrained_layout=True)
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(METHOD_ORDER))
    for offset, method in zip(offsets, METHOD_ORDER):
        values = [compact["stress_sets"][s]["methods"].get(method, {}).get("stress_accuracy", np.nan) * 100 for s in stress_names]
        ax.bar(x + offset, values, width, label=METHOD_LABELS[method])
    ax.set_xticks(x)
    ax.set_xticklabels(stress_names, rotation=15, ha="right")
    ax.set_ylabel("Stress test accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("P02 default-trained methods under controlled stress sets")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_gap_bars(compact: dict[str, Any], out: Path) -> None:
    stress_names = list(compact["stress_sets"].keys())
    x = np.arange(len(stress_names))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5.6), constrained_layout=True)
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(METHOD_ORDER))
    for offset, method in zip(offsets, METHOD_ORDER):
        values = [compact["stress_sets"][s]["methods"].get(method, {}).get("generalization_gap", np.nan) * 100 for s in stress_names]
        ax.bar(x + offset, values, width, label=METHOD_LABELS[method])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(stress_names, rotation=15, ha="right")
    ax.set_ylabel("Default − stress accuracy gap (percentage points)")
    ax.set_title("P02 generalization gaps from the default `[0°,60°]` training distribution")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _find_sample_indices(labels: np.ndarray, class_names: list[str]) -> list[int]:
    indices = []
    for class_name in class_names:
        label = ACTIVITY_LABELS.index(class_name)
        matches = np.where(labels == label)[0]
        if len(matches) == 0:
            continue
        indices.append(int(matches[0]))
    return indices


def plot_sample_grid(data_dirs: dict[str, Path], out: Path) -> None:
    rows = []
    for display_name, data_dir in data_dirs.items():
        path = data_dir / "har_test.h5"
        if not path.exists():
            continue
        with h5py.File(path, "r") as f:
            y = f["y"][:]
            indices = _find_sample_indices(y, DISPLAY_CLASSES)
            xs = [f["x"][i, 0] for i in indices]
            rows.append((display_name, indices, xs))
    if not rows:
        return

    nrows = len(rows)
    ncols = len(DISPLAY_CLASSES)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.65 * nrows), squeeze=False, constrained_layout=True)
    for r, (display_name, indices, xs) in enumerate(rows):
        for c, class_name in enumerate(DISPLAY_CLASSES):
            ax = axes[r, c]
            if c >= len(xs):
                ax.axis("off")
                continue
            ax.imshow(xs[c], aspect="auto", origin="lower", cmap="magma")
            if r == 0:
                ax.set_title(class_name)
            if c == 0:
                ax.set_ylabel(display_name)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Example P02 target-range micro-Doppler spectrograms")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _normalize_confusion(matrix: list[list[int]]) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    denom = arr.sum(axis=1, keepdims=True)
    return np.divide(arr, np.maximum(denom, 1.0))


def plot_confusion(stress_name: str, stress_dir: Path, out_dir: Path) -> list[Path]:
    paths = []
    for method, file_name in CONFUSION_FILES.items():
        result_path = stress_dir / file_name
        if not result_path.exists():
            continue
        payload = load_json(result_path)
        matrix = (
            payload.get("eval_confusion_matrix")
            or payload.get("test_confusion_matrix")
            or payload.get("confusion_matrix")
        )
        if matrix is None:
            continue
        norm = _normalize_confusion(matrix)
        fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{stress_name} — {METHOD_LABELS[method]} confusion (row-normalized)")
        ax.set_xticks(range(len(ACTIVITY_LABELS)))
        ax.set_yticks(range(len(ACTIVITY_LABELS)))
        ax.set_xticklabels(ACTIVITY_LABELS, rotation=40, ha="right")
        ax.set_yticklabels(ACTIVITY_LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                value = norm[i, j]
                if value >= 0.01 or i == j:
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.55 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        out = out_dir / f"confusion_{stress_name}_{method}.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        paths.append(out)
    return paths


def render_report(compact: dict[str, Any], out_dir: Path, figure_paths: list[Path]) -> str:
    stress_names = list(compact["stress_sets"].keys())
    lines = [
        "# P02 Stress-Generalization Visual Report",
        "",
        "P02 here means **target-range micro-Doppler classification** generated from the P02-only scatterer model. It is range-compressed/matched-filter style target-range extraction, not a full raw FMCW dechirp cube task.",
        "",
        "The default model/checkpoint training distribution is schema v6 with absolute aspect `[0°,60°]`. The stress sets below measure how those same trained baselines generalize to held-out aspect, SNR, and range regimes.",
        "",
        "## Summary figures",
        "",
        f"![Stress accuracy]({Path('accuracy_bars.png')})",
        "",
        f"![Generalization gaps]({Path('gap_bars.png')})",
        "",
        f"![Spectrogram examples]({Path('sample_spectrogram_grid.png')})",
        "",
        "## Accuracy table",
        "",
        "| Stress set | Method | Default acc | Stress acc | Gap | Worst class drop |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for stress_name in stress_names:
        methods = compact["stress_sets"][stress_name]["methods"]
        for method in METHOD_ORDER:
            result = methods.get(method)
            if not result:
                continue
            worst = result.get("worst_per_class_drop")
            worst_txt = "n/a"
            if worst:
                worst_txt = f"{worst['class_name']} ({worst['drop'] * 100:.1f} pp)"
            lines.append(
                f"| {stress_name} | {METHOD_LABELS[method]} | "
                f"{result['default_accuracy'] * 100:.2f}% | "
                f"{result['stress_accuracy'] * 100:.2f}% | "
                f"{result['generalization_gap'] * 100:.2f} pp | {worst_txt} |"
            )
    lines.extend([
        "",
        "## Confusion matrices",
        "",
    ])
    for path in figure_paths:
        if path.name.startswith("confusion_"):
            lines.append(f"![{path.stem}]({path.name})")
            lines.append("")
    lines.extend([
        "## Teaching notes",
        "",
        "- Aspect stress is physically meaningful here because radial Doppler scales with the aspect projection; signed aspect is symmetric under the current 2-D model, so absolute aspect is the stored convention.",
        "- Low-SNR stress tests whether morphology/temporal cues remain separable once the micro-Doppler spectrogram is noisier.",
        "- Far-range stress is secondary because the generator explicitly samples labelled SNR; a weak range effect should be reported as a simulator-design consequence, not as empirical range invariance.",
    ])
    return "\n".join(lines) + "\n"


def render_html_report(compact: dict[str, Any], figure_paths: list[Path]) -> str:
    """Return a self-contained HTML shell that references generated PNG files."""
    rows = []
    for stress_name in compact["stress_sets"]:
        methods = compact["stress_sets"][stress_name]["methods"]
        for method in METHOD_ORDER:
            result = methods.get(method)
            if not result:
                continue
            worst = result.get("worst_per_class_drop")
            worst_txt = "n/a"
            if worst:
                worst_txt = f"{worst['class_name']} ({worst['drop'] * 100:.1f} pp)"
            rows.append(
                "<tr>"
                f"<td>{html.escape(stress_name)}</td>"
                f"<td>{METHOD_LABELS[method]}</td>"
                f"<td>{result['default_accuracy'] * 100:.2f}%</td>"
                f"<td>{result['stress_accuracy'] * 100:.2f}%</td>"
                f"<td>{result['generalization_gap'] * 100:.2f} pp</td>"
                f"<td>{html.escape(worst_txt)}</td>"
                "</tr>"
            )

    data_rows = []
    for stress_name, payload in compact["stress_sets"].items():
        data = payload["data_test_summary"]
        data_rows.append(
            "<tr>"
            f"<td>{html.escape(stress_name)}</td>"
            f"<td>{data['x_shape'][0]}</td>"
            f"<td>{data['aspect_angle_minmax_deg'][0]:.2f}–"
            f"{data['aspect_angle_minmax_deg'][1]:.2f}°</td>"
            f"<td>{data['snr_minmax_db'][0]:.2f}–{data['snr_minmax_db'][1]:.2f} dB</td>"
            f"<td>{data['range_minmax_m'][0]:.2f}–{data['range_minmax_m'][1]:.2f} m</td>"
            f"<td>{data['min_doppler_alias_margin_mps']:.3f} m/s</td>"
            "</tr>"
        )

    confusion_figures = "\n".join(
        f'<figure><img src="{path.name}" alt="{path.stem}">'
        f"<figcaption>{html.escape(path.stem)}</figcaption></figure>"
        for path in figure_paths
        if path.name.startswith("confusion_")
    )
    style = """
:root { color-scheme: light; }
body { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2rem auto; max-width: 1180px; padding: 0 1rem; line-height: 1.55; color: #18212f; }
h1,h2,h3 { line-height: 1.2; }
.hero { padding: 1.2rem 1.4rem; background: #eef6ff; border: 1px solid #cfe4ff; border-radius: 14px; }
.note { background:#fff8e6; border:1px solid #f0dc9a; border-radius: 10px; padding: .8rem 1rem; }
table { width: 100%; border-collapse: collapse; margin: 1rem 0 1.5rem; font-size: .94rem; }
th, td { border-bottom: 1px solid #d8dee9; padding: .55rem .65rem; text-align: left; }
th { background: #f6f8fb; }
img { max-width: 100%; height: auto; border: 1px solid #d8dee9; border-radius: 10px; background: white; }
.figure-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 1rem; align-items: start; }
figure { margin: 0; }
figcaption { font-size: .82rem; color: #566070; margin-top: .3rem; }
.small { color: #566070; font-size: .92rem; }
"""
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>P02 Stress-Generalization Visual Report</title>
<style>{style}</style>
</head>
<body>
<section class="hero">
<h1>P02 Stress-Generalization Visual Report</h1>
<p><strong>결론:</strong> 기본 IID P02는 neural/feature 모두 거의 포화지만, held-out oblique aspect(60–80°)에서는 handcrafted feature가 크게 무너지고 neural 모델도 12–14 pp 정도 drop이 발생한다. Low-SNR에서는 handcrafted feature만 크게 하락하고 TinyCNN/ResNet18은 거의 유지된다.</p>
<p class="small">생성 기준: schema v6, target-range micro-Doppler, P02-only scatterer model. Raw FMCW dechirp cube가 아니라 range-compressed/matched-filter style target-range extraction 기반이다.</p>
</section>

<h2>Accuracy / gap overview</h2>
<div class="figure-grid">
<figure><img src="accuracy_bars.png" alt="Stress accuracy bars"><figcaption>Stress-set accuracy by method.</figcaption></figure>
<figure><img src="gap_bars.png" alt="Generalization gap bars"><figcaption>Default minus stress accuracy gap.</figcaption></figure>
</div>

<h2>Example micro-Doppler spectrograms</h2>
<figure><img src="sample_spectrogram_grid.png" alt="Sample spectrogram grid"><figcaption>Default and stress-set examples for walk/run/wave/idle.</figcaption></figure>

<h2>Accuracy table</h2>
<table><thead><tr><th>Stress set</th><th>Method</th><th>Default acc</th><th>Stress acc</th><th>Gap</th><th>Worst class drop</th></tr></thead><tbody>
{''.join(rows)}
</tbody></table>

<h2>Data checks</h2>
<table><thead><tr><th>Stress set</th><th>Test N</th><th>Aspect min-max</th><th>SNR min-max</th><th>Range min-max</th><th>Min alias margin</th></tr></thead><tbody>
{''.join(data_rows)}
</tbody></table>

<div class="note">
<h3>Teaching interpretation</h3>
<ul>
<li>Aspect stress is the strongest and physically interpretable shift: radial Doppler shrinks under oblique aspect, so walk/run morphology becomes less like the default training distribution.</li>
<li>Low-SNR reveals a useful network-vs-handcrafted contrast: RBF/logreg feature baselines lose ~15–16 pp, while CNN/ResNet remain nearly saturated.</li>
<li>Far-range stress is intentionally muted because labelled SNR is explicitly sampled; this should be taught as a simulator-design consequence, not as proof of range-invariant sensing.</li>
<li>Aspect angle is stored as absolute degrees because the current 2-D radial projection is symmetric for signed aspect.</li>
</ul>
</div>

<h2>Confusion matrices</h2>
<div class="figure-grid">
{confusion_figures}
</div>

<p class="small">Artifacts: compact summary JSON, result JSON files, PNG figures, and this HTML report are generated by the stress-evaluation scripts.</p>
</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Make P02 stress-eval figures and report")
    ap.add_argument("--summary", default="artifacts/stress_eval/p02_stress_comparison_summary_compact.json")
    ap.add_argument("--stress_root", default="artifacts/stress_eval")
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--out_dir", default="artifacts/stress_eval/report")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    stress_root = Path(args.stress_root)
    data_root = Path(args.data_root)
    out_dir = ensure_out(Path(args.out_dir))
    compact = load_json(summary_path)

    figure_paths: list[Path] = []
    acc_path = out_dir / "accuracy_bars.png"
    gap_path = out_dir / "gap_bars.png"
    sample_path = out_dir / "sample_spectrogram_grid.png"
    plot_accuracy_bars(compact, acc_path)
    plot_gap_bars(compact, gap_path)
    figure_paths.extend([acc_path, gap_path])

    data_dirs = {"default": data_root / "data"}
    for stress_name in compact["stress_sets"]:
        data_dirs[stress_name] = data_root / stress_name
    plot_sample_grid(data_dirs, sample_path)
    if sample_path.exists():
        figure_paths.append(sample_path)

    for stress_name in compact["stress_sets"]:
        figure_paths.extend(plot_confusion(stress_name, stress_root / stress_name, out_dir))

    report_md = out_dir / "p02_stress_visual_report.md"
    report_md.write_text(render_report(compact, out_dir, figure_paths))
    report_html = out_dir / "p02_stress_visual_report.html"
    report_html.write_text(render_html_report(compact, figure_paths))
    print(f"saved {report_md}")
    print(f"saved {report_html}")
    for path in figure_paths:
        print(f"saved {path}")


if __name__ == "__main__":
    main()
