#!/usr/bin/env python3
"""Build a compact shareable P03 lecture bundle index."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent

DEFAULT_LINKS = {
    "main": "https://remilab.cnu.ac.kr/share/576d311ccc45/p03_main_result_report.html",
    "doa": "https://remilab.cnu.ac.kr/share/639f797adc4b/p03_doa_diagnostics.html",
    "angular": "https://remilab.cnu.ac.kr/share/519e2020fc77/p03_angular_resolution_report.html",
    "range": "https://remilab.cnu.ac.kr/share/576d311ccc45/p03_resolution_report.html",
    "ego": "https://remilab.cnu.ac.kr/share/e470aa251013/p03_ego_motion_error_report.html",
    "offgrid": "https://remilab.cnu.ac.kr/share/576d311ccc45/p03_offgrid_appendix.html",
}


def fmt(v: object, digits: int = 3) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def load_rows(config_path: Path) -> list[dict]:
    if not config_path.exists():
        return []
    data = json.loads(config_path.read_text())
    return data.get("metrics_rows", [])


def make_report(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(args.main_config))
    metric_rows = "\n".join(
        "<tr>"
        f"<td>{r['method']}</td>"
        f"<td>{fmt(r['mae_deg'])}</td>"
        f"<td>{fmt(r['within_2deg_acc'])}</td>"
        f"<td>{fmt(r['pc_iou'])}</td>"
        f"<td>{fmt(r['point_error_mean_m'])}</td>"
        "</tr>"
        for r in rows
    )
    links = DEFAULT_LINKS.copy()
    for key in links:
        arg_name = f"{key}_url"
        if hasattr(args, arg_name) and getattr(args, arg_name):
            links[key] = getattr(args, arg_name)
    offgrid_link = (
        f"<a href=\"{links['offgrid']}\">Open off-grid raster appendix</a>"
        if links.get("offgrid")
        else "Generate locally with <code>make_offgrid_appendix.py</code>."
    )
    html = f"""<!doctype html>
<html lang=\"ko\"><head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>P03 Lecture Bundle — Radar Mapping as DoA Quality Testbed</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 32px; color: #111827; line-height: 1.55; }}
main {{ max-width: 1100px; margin: auto; }}
.note {{ background: #eff6ff; border-left: 4px solid #2563eb; padding: 12px 16px; margin: 18px 0; }}
.warn {{ background: #fff7ed; border-left: 4px solid #f97316; padding: 12px 16px; margin: 18px 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 18px 0 28px; }}
th, td {{ border: 1px solid #d1d5db; padding: 7px 9px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #f3f4f6; }}
a {{ color: #1d4ed8; }}
.card {{ border: 1px solid #d1d5db; border-radius: 10px; padding: 14px 16px; margin: 12px 0; }}
code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }}
</style></head><body><main>
<h1>P03 Lecture Bundle — Radar Mapping as DoA Quality Testbed</h1>
<div class=\"note\"><strong>Core thesis.</strong> DoA accuracy is not only an angle metric. With known ego motion, range-bearing detections are projected into a world-frame point-cloud / probabilistic map, so DoA quality becomes visible as map quality.</div>
<div class=\"warn\"><strong>Controlled setting.</strong> The main result isolates DoA by using simulator-exact range, perfect ego pose, and the same RD-selected antenna-vector detections for every method.</div>
<h2>Canonical main result</h2>
<p><a href=\"{links['main']}\">Open canonical uniform-grid main report</a></p>
<table><thead><tr><th>Method</th><th>DoA MAE [deg]</th><th>≤2° acc.</th><th>Point-grid IoU</th><th>Mean point error [m]</th></tr></thead><tbody>{metric_rows}</tbody></table>
<p><strong>Interpretation:</strong> MUSIC is currently the strongest clean selected-vector reference. RadarCubeDoANet is sub-degree and close to MUSIC in the map projection, but does not beat MUSIC. The coarse native angle FFT is the weak/failure baseline.</p>
<h2>Teaching sequence</h2>
<div class=\"card\"><h3>1. Main DoA-to-map result</h3><p><a href=\"{links['main']}\">Open main map result</a></p><p>Start here: single-frame vs ego-motion accumulated map, plus oracle/MUSIC/DoANet/coarse FFT comparison on the uniform square-cell grid.</p></div>
<div class=\"card\"><h3>2. DoA-only diagnostics</h3><p><a href=\"{links['doa']}\">Open DoA diagnostics</a></p><p>Then step back to estimator-level accuracy: which method estimates the angle accurately before map projection?</p></div>
<div class=\"card\"><h3>3. Angular / cross-range projection appendix</h3><p><a href=\"{links['angular']}\">Open angular projection appendix</a></p><p>Use this to explain <code>lateral error ≈ R·Δθ</code>. It is a projection-sensitivity probe, not a simultaneous two-target super-resolution benchmark.</p></div>
<div class=\"card\"><h3>4. Range-resolution appendix</h3><p><a href=\"{links['range']}\">Open range-resolution appendix</a></p><p>Show 50 MHz as the low-resolution stress condition, 200 MHz as the practical baseline, and first-hit occlusion as the wall-scatterer visibility rule.</p></div>
<div class=\"card\"><h3>5. Off-grid raster appendix</h3><p>{offgrid_link}</p><p>Use this as the grid-metric caveat: GT DoA/range and ego pose are fixed, and sub-cell scene shifts show that grid IoU is a raster/threshold metric while continuous point error remains zero.</p></div>
<div class=\"card\"><h3>6. Ego-motion-error appendix</h3><p><a href=\"{links['ego']}\">Open ego-motion-error appendix</a></p><p>Close with the separate odometry/calibration issue: GT DoA/range are fixed and only pose/yaw/drift are perturbed.</p></div>
<h2>Slide caveats to keep visible</h2>
<ul>
<li>Main report isolates DoA; range-bin and ego-pose errors are appendix-only.</li>
<li>Canonical map grid: x=[-20,20] m, y=[0,40] m, 128×128, cell=0.3125 m.</li>
<li>OGM IoU is secondary. Point-grid IoU and point error are the clearer teaching metrics.</li>
<li>DoA diagnostics and canonical map result answer different questions: estimator accuracy vs map impact.</li>
<li>Four held-out canonical scenes are enough for lecture evidence, not broad generalization claims.</li>
</ul>
</main></body></html>"""
    html_path = out_dir / "p03_lecture_bundle.html"
    html_path.write_text(html, encoding="utf-8")
    return {"html": str(html_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build P03 lecture bundle index")
    parser.add_argument("--main_config", default=str(BASE / "artifacts" / "main_result_canonical_uniform128" / "p03_main_result_config.json"))
    parser.add_argument("--out_dir", default=str(BASE / "artifacts" / "lecture_bundle"))
    parser.add_argument("--main_url", default=None)
    parser.add_argument("--doa_url", default=None)
    parser.add_argument("--angular_url", default=None)
    parser.add_argument("--range_url", default=None)
    parser.add_argument("--ego_url", default=None)
    parser.add_argument("--offgrid_url", default=None)
    args = parser.parse_args()
    print(json.dumps(make_report(args), indent=2))


if __name__ == "__main__":
    main()
