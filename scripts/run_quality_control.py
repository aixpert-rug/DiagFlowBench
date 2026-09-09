#!/usr/bin/env python3
"""Run quality control over all generated conversations.

Computes:
  - Self-BLEU diversity score (global and per-graph)
  - Edge-label consistency (operator utterances vs. expected branch labels)

Writes a quality_report.json to the conversations directory.

Prerequisites:
  - Conversations must be generated first (run_generation.py or run_batch_generation.py)
  - nltk punkt tokenizer must be available: python -c "import nltk; nltk.download('punkt')"

Usage:
    python scripts/run_quality_control.py
    python scripts/run_quality_control.py --conversations-dir path/to/conversations
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONVERSATIONS_DIR, SELF_BLEU_THRESHOLD, EDGE_LABEL_CONSISTENCY_THRESHOLD
from src.conversations.quality import run_quality_control


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quality control for DIAGFLOWBENCH generated conversations"
    )
    parser.add_argument(
        "--conversations-dir",
        type=str,
        default=None,
        help=f"Path to conversations directory (default: {CONVERSATIONS_DIR})",
    )
    args = parser.parse_args()

    conversations_dir = Path(args.conversations_dir) if args.conversations_dir else CONVERSATIONS_DIR

    if not conversations_dir.exists():
        print(f"ERROR: Conversations directory not found: {conversations_dir}")
        print("Run scripts/run_generation.py or scripts/run_batch_generation.py first.")
        sys.exit(1)

    conv_files = list(conversations_dir.glob("*.json"))
    # Exclude the report itself if it already exists
    conv_files = [f for f in conv_files if f.name != "quality_report.json"]
    if not conv_files:
        print(f"ERROR: No conversation JSON files found in {conversations_dir}")
        sys.exit(1)

    print("DIAGFLOWBENCH — Quality Control")
    print(f"Conversations directory: {conversations_dir}")
    print(f"Files to evaluate:       {len(conv_files)}")
    print(f"Self-BLEU threshold:     < {SELF_BLEU_THRESHOLD} (lower = more diverse)")
    print(f"Consistency threshold:   >= {EDGE_LABEL_CONSISTENCY_THRESHOLD}")
    print()

    try:
        report = run_quality_control(conversations_dir=conversations_dir)
    except ModuleNotFoundError as exc:
        if "nltk" in str(exc):
            print("ERROR: nltk is required for Self-BLEU computation.")
            print("Install it and download the punkt tokenizer:")
            print("  pip install nltk")
            print("  python -c \"import nltk; nltk.download('punkt')\"")
            sys.exit(1)
        raise

    report_path = conversations_dir / "quality_report.json"

    print("=" * 60)
    print("QUALITY REPORT SUMMARY")
    print("=" * 60)

    bleu = report["self_bleu"]
    bleu_status = "PASS" if bleu["passed"] else "FAIL"
    print(f"Self-BLEU (global):    {bleu['global']:.4f}  [{bleu_status}]  (threshold < {bleu['threshold']})")

    cons = report["edge_label_consistency"]
    cons_status = "PASS" if cons["pass_rate"] >= EDGE_LABEL_CONSISTENCY_THRESHOLD else "FAIL"
    print(
        f"Edge consistency:      {cons['average_score']:.4f}  [{cons_status}]  "
        f"({cons['passed_conversations']}/{cons['total_conversations']} conversations passed)"
    )

    print(f"\nFull report written to: {report_path}")

    if bleu_status == "FAIL" or cons_status == "FAIL":
        print("\nWARNING: One or more quality checks failed. Review quality_report.json for details.")
        sys.exit(2)


if __name__ == "__main__":
    main()
