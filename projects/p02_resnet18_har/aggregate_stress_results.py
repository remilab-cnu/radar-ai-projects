#!/usr/bin/env python3
"""Aggregate P02 default and stress-generalization result JSONs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METHODS = {
    "feature_logreg": {
        "label": "Handcrafted LogReg",
        "file": "feature_logreg.json",
        "default_key": "feature_logreg",
    },
    "feature_rbf_svm_10k": {
        "label": "Handcrafted RBF SVM (10k)",
        "file": "feature_rbf_svm_10k.json",
        "default_key": "feature_rbf_svm_10k",
    },
    "tiny_cnn": {
        "label": "TinyCNN",
        "file": "tiny_cnn.json",
        "default_key": "tiny_cnn",
    },
    "resnet18": {
        "label": "ResNet18",
        "file": "resnet18.json",
        "default_key": "resnet18",
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def accuracy(payload: dict[str, Any]) -> float | None:
    for key in ("eval_accuracy", "test_accuracy", "accuracy"):
        value = payload.get(key)
        if value is not None:
            return float(value)
    return None


def per_class(payload: dict[str, Any]) -> dict[str, float]:
    value = payload.get("eval_per_class") or payload.get("test_per_class") or payload.get("per_class") or {}
    return {str(k): float(v) for k, v in value.items()}


def confusion(payload: dict[str, Any]) -> list[list[int]] | None:
    value = (
        payload.get("eval_confusion_matrix")
        or payload.get("test_confusion_matrix")
        or payload.get("confusion_matrix")
    )
    return value


def split_summary(data_summary: dict[str, Any]) -> dict[str, Any]:
    # Prefer test split because that is the evaluation split.
    return data_summary.get("har_test.h5", data_summary)


def default_accuracy(default_summary: dict[str, Any], method_key: str) -> float | None:
    payload = default_summary.get(METHODS[method_key]["default_key"], {})
    return accuracy(payload)


def default_per_class(default_summary: dict[str, Any], method_key: str) -> dict[str, float]:
    payload = default_summary.get(METHODS[method_key]["default_key"], {})
    return per_class(payload)


def worst_class_drop(default_pc: dict[str, float], stress_pc: dict[str, float]) -> dict[str, Any] | None:
    common = sorted(set(default_pc) & set(stress_pc))
    if not common:
        return None
    drops = {cls: default_pc[cls] - stress_pc[cls] for cls in common}
    cls = max(drops, key=drops.get)
    return {
        "class_name": cls,
        "default_accuracy": default_pc[cls],
        "stress_accuracy": stress_pc[cls],
        "drop": drops[cls],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# P02 Stress-Generalization Summary",
        "",
        "Default training distribution: schema v6, absolute aspect `[0°, 60°]`, target-range micro-Doppler from the P02-only scatterer model.",
        "",
        "## Accuracy and generalization gap",
        "",
        "| Stress set | Method | Default test acc | Stress test acc | Gap | Worst class drop |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for stress_name, stress in summary["stress_sets"].items():
        for method_key, result in stress["methods"].items():
            worst = result.get("worst_per_class_drop")
            if worst:
                worst_txt = f"{worst['class_name']} ({worst['drop']*100:.1f} pp)"
            else:
                worst_txt = "n/a"
            lines.append(
                f"| {stress_name} | {METHODS[method_key]['label']} | "
                f"{result['default_accuracy']*100:.2f}% | "
                f"{result['stress_accuracy']*100:.2f}% | "
                f"{result['generalization_gap']*100:.2f} pp | {worst_txt} |"
            )
    lines.extend([
        "",
        "## Stress-set data checks",
        "",
        "| Stress set | Test samples | Aspect range/meta | Aspect min-max | SNR min-max | Range min-max | Min alias margin |",
        "| --- | ---: | --- | --- | --- | --- | ---: |",
    ])
    for stress_name, stress in summary["stress_sets"].items():
        data = split_summary(stress.get("data_summary", {}))
        lines.append(
            f"| {stress_name} | {data.get('x_shape', ['?'])[0]} | "
            f"{data.get('aspect_angle_range_deg')} | "
            f"{data.get('aspect_angle_minmax_deg')} | "
            f"{data.get('snr_minmax_db')} | "
            f"{data.get('range_minmax_m')} | "
            f"{float(data.get('min_doppler_alias_margin_mps', 0.0)):.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation notes",
        "",
        "- P02 uses target-range micro-Doppler extracted from a P02-only scatterer model; it is not a full raw FMCW dechirp cube task.",
        "- Aspect uses an absolute-angle convention because the current 2-D radial projection is signed-aspect symmetric.",
        "- Far-range stress is expected to be muted when the generator explicitly samples labelled SNR; this is a simulator-design observation, not proof of range-invariant sensing.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate P02 stress-eval JSON files")
    ap.add_argument("--default_summary", default="artifacts/full_eval/p02_default_comparison_summary_compact.json")
    ap.add_argument("--stress_root", default="artifacts/stress_eval")
    ap.add_argument("--out_json", default="artifacts/stress_eval/p02_stress_comparison_summary.json")
    ap.add_argument("--out_compact", default="artifacts/stress_eval/p02_stress_comparison_summary_compact.json")
    ap.add_argument("--out_md", default="artifacts/stress_eval/p02_stress_comparison_summary.md")
    args = ap.parse_args()

    default_path = Path(args.default_summary)
    stress_root = Path(args.stress_root)
    default_summary = load_json(default_path)

    summary: dict[str, Any] = {
        "kind": "p02_stress_generalization_summary",
        "default_summary": str(default_path),
        "stress_root": str(stress_root),
        "methods": {key: {"label": spec["label"], "default_key": spec["default_key"]} for key, spec in METHODS.items()},
        "stress_sets": {},
    }

    for stress_dir in sorted(p for p in stress_root.iterdir() if p.is_dir()):
        stress_name = stress_dir.name
        data_path = stress_dir / "data_summary.json"
        data_summary = load_json(data_path) if data_path.exists() else {}
        stress_payload: dict[str, Any] = {"data_summary": data_summary, "methods": {}}
        for method_key, spec in METHODS.items():
            result_path = stress_dir / spec["file"]
            if not result_path.exists():
                continue
            payload = load_json(result_path)
            stress_acc = accuracy(payload)
            base_acc = default_accuracy(default_summary, method_key)
            if stress_acc is None or base_acc is None:
                continue
            stress_pc = per_class(payload)
            base_pc = default_per_class(default_summary, method_key)
            stress_payload["methods"][method_key] = {
                "label": spec["label"],
                "result_path": str(result_path),
                "default_accuracy": base_acc,
                "stress_accuracy": stress_acc,
                "generalization_gap": base_acc - stress_acc,
                "per_class": stress_pc,
                "confusion_matrix": confusion(payload),
                "worst_per_class_drop": worst_class_drop(base_pc, stress_pc),
            }
        summary["stress_sets"][stress_name] = stress_payload

    compact = {
        "kind": summary["kind"],
        "default_summary": summary["default_summary"],
        "stress_sets": {},
    }
    for stress_name, stress in summary["stress_sets"].items():
        compact["stress_sets"][stress_name] = {
            "data_test_summary": split_summary(stress.get("data_summary", {})),
            "methods": {
                method_key: {
                    "label": result["label"],
                    "default_accuracy": result["default_accuracy"],
                    "stress_accuracy": result["stress_accuracy"],
                    "generalization_gap": result["generalization_gap"],
                    "worst_per_class_drop": result["worst_per_class_drop"],
                }
                for method_key, result in stress["methods"].items()
            },
        }

    out_json = Path(args.out_json)
    out_compact = Path(args.out_compact)
    out_md = Path(args.out_md)
    for path in [out_json, out_compact, out_md]:
        path.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    out_compact.write_text(json.dumps(compact, indent=2))
    out_md.write_text(render_markdown(summary))

    print(json.dumps(compact, indent=2))
    print(f"saved {out_json}")
    print(f"saved {out_compact}")
    print(f"saved {out_md}")


if __name__ == "__main__":
    main()
