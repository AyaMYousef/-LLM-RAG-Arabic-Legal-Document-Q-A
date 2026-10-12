"""
Faithfulness alert script for the Arabic Legal RAG pipeline.

Reads data/evaluation/ragas_results.json and triggers an alert if
faithfulness < THRESHOLD (default 0.80).

Alert channels:
  - Terminal: prints a visible WARNING block
  - File: appends a timestamped entry to reports/alerts.log
  - Desktop: optional toast notification via plyer

Exit codes:
    0  faithfulness >= threshold (no alert)
    1  faithfulness < threshold or results file missing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_FILE = (
    PROJECT_ROOT / "data" / "evaluation" / "ragas_results.json"
)
ALERTS_LOG = PROJECT_ROOT / "reports" / "alerts.log"

DEFAULT_THRESHOLD = 0.80

logger = logging.getLogger(__name__)


def _desktop_notify(title: str, message: str) -> None:
    """Fire a desktop toast notification if plyer is installed."""
    try:
        from plyer import notification  # type: ignore[import-not-found]
    except ImportError:
        logger.debug("Desktop notification skipped: plyer is not installed.")
        return

    try:
        notification.notify(
            title=title,
            message=message,
            app_name="Arabic Legal RAG",
            timeout=10,
        )
    except (RuntimeError, OSError) as exc:
        logger.warning("Desktop notification failed: %s", exc)


def check_faithfulness(results_file: Path, threshold: float) -> int:
    """Check faithfulness and fire alerts if the score is below threshold."""
    if not results_file.exists():
        print(f"[alert] Results file not found: {results_file}")
        return 1

    with results_file.open(encoding="utf-8") as f:
        data = json.load(f)

    faithfulness = data["metrics"]["faithfulness"]
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    if faithfulness >= threshold:
        print(
            f"[alert] Faithfulness OK: {faithfulness:.4f} "
            f"(threshold {threshold})"
        )
        return 0

    alert_msg = (
        f"FAITHFULNESS ALERT — {timestamp}\n"
        f"  Score     : {faithfulness:.4f}\n"
        f"  Threshold : {threshold}\n"
        f"  Drop      : {threshold - faithfulness:.4f}\n"
        "  Action    : Review recent changes to chunking, "
        "retrieval, or generation config."
    )

    print("\n" + "=" * 60)
    print("WARNING: RAGAS FAITHFULNESS BELOW THRESHOLD")
    print("=" * 60)
    print(alert_msg)
    print("=" * 60 + "\n")

    ALERTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ALERTS_LOG.open("a", encoding="utf-8") as log:
        log.write(alert_msg + "\n" + "-" * 60 + "\n")
    print(f"[alert] Alert written to {ALERTS_LOG}")

    _desktop_notify(
        title="RAG Faithfulness Alert",
        message=(
            f"Faithfulness {faithfulness:.4f} is below "
            f"threshold {threshold}. Check reports/alerts.log."
        ),
    )

    return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Alert if RAGAS faithfulness is below threshold."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Faithfulness threshold (default {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_FILE,
        help="Path to ragas_results.json",
    )
    args = parser.parse_args()

    sys.exit(check_faithfulness(args.results, args.threshold))


if __name__ == "__main__":
    main()

