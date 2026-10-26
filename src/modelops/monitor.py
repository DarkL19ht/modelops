# Create the monitoring report generator
"""CLI runner for feature-distribution monitoring and reporting."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.modelops.drift import calculate_drift


ROOT = Path(__file__).resolve().parents[2]


def save_reports(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save a machine-readable report.
    (output_dir / "drift_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    # Build a human-readable Markdown report.
    rows = [
        "# ModelOps — Data Drift Report",
        "",
        f"- Reference observations: **{report['reference_rows']:,}**",
        f"- Incoming observations: **{report['incoming_rows']:,}**",
        f"- PSI alert threshold: **{report['psi_threshold']:.2f}**",
        f"- Features flagged: **{len(report['drifted_features'])} of {len(report['features'])}**",
        "",
        "| Feature | Type | PSI | Status | Reference mean | Incoming mean | KS statistic |",
        "|---|---|---:|---|---:|---:|---:|",
    ]

    for result in report["features"]:
        ks = (
            "—"
            if result["ks_statistic"] is None
            else f"{result['ks_statistic']:.3f}"
        )

        rows.append(
            f"| {result['feature']} "
            f"| {result['type']} "
            f"| {result['psi']:.3f} "
            f"| {result['status']} "
            f"| {result['reference_mean']:.2f} "
            f"| {result['incoming_mean']:.2f} "
            f"| {ks} |"
        )

    rows.extend([
        "",
        "## Interpretation",
        "",
        "Drift flags compare input-feature distributions; "
        "they do not establish that model performance "
        "has deteriorated.",
        "",
        "PSI at or above the configured threshold triggers "
        "a drift flag. PSI depends on reference-based "
        "binning and sample size. The KS statistic is "
        "descriptive for discrete count features; "
        "its p-value assumptions are most appropriate "
        "for continuous observations.",
        "",
        "The incoming dataset is synthetically shifted by "
        "design. Results are a demonstration, not a "
        "real-world performance claim.",
        "",
    ])

    (output_dir / "drift_report.md").write_text(
        "\n".join(rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare reference and incoming "
            "feature distributions"
        )
    )

    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "data/generated/reference.csv",
    )

    parser.add_argument(
        "--incoming",
        type=Path,
        default=ROOT / "data/generated/incoming.csv",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports",
    )

    args = parser.parse_args()

    report = calculate_drift(
        pd.read_csv(args.reference),
        pd.read_csv(args.incoming),
        threshold=args.threshold,
    )

    save_reports(
        report,
        args.output_dir,
    )

    print(
        f"Analysed {len(report['features'])} features"
    )

    print(
        "Drifted: "
        + (
            ", ".join(report["drifted_features"])
            or "none"
        )
    )

    print(
        f"Reports saved in {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()