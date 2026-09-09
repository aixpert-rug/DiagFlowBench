#!/usr/bin/env python3
"""Run the conversation generation pipeline using the Anthropic Batch API.

Three phases, run sequentially (each saves state to batch_state.json):
  submit, collect-pass1, collect-pass2.

Usage:
    python scripts/run_batch_generation.py --phase submit
    python scripts/run_batch_generation.py --phase collect-pass1
    python scripts/run_batch_generation.py --phase collect-pass2
    python scripts/run_batch_generation.py --phase collect-pass1 --batch-id msgbatch_xxx
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conversations.batch_generator import (
    submit_pass1_batch,
    collect_pass1_and_submit_pass2,
    collect_pass2_and_assemble,
)
from src.config import PATHS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DIAGFLOWBENCH — Anthropic Batch API generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=["submit", "collect-pass1", "collect-pass2"],
        help=(
            "submit: build & submit Pass-1 generation batch | "
            "collect-pass1: poll Pass-1, submit Pass-2 batch | "
            "collect-pass2: poll Pass-2, assemble conversations"
        ),
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Anthropic batch ID (optional; loaded from batch_state.json if omitted)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to manifest JSON (default: DiagFlowBench_Dataset/paths/manifest.json)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest) if args.manifest else PATHS_DIR / "manifest.json"

    if args.phase == "submit":
        if not manifest_path.exists():
            print(f"ERROR: Manifest not found at {manifest_path}")
            print("Run scripts/run_path_enumeration.py first.")
            sys.exit(1)
        batch_id = submit_pass1_batch(manifest_path=manifest_path)
        print(f"\nPhase 1 complete. Batch ID: {batch_id}")
        print("Run again with --phase collect-pass1 when the batch has processed.")

    elif args.phase == "collect-pass1":
        batch_id = collect_pass1_and_submit_pass2(batch_id=args.batch_id)
        print(f"\nPhase 2 complete. Pass-2 Batch ID: {batch_id}")
        print("Run again with --phase collect-pass2 when the batch has processed.")

    elif args.phase == "collect-pass2":
        output_files = collect_pass2_and_assemble(batch_id=args.batch_id)
        print(f"\nPhase 3 complete. {len(output_files)} conversation files written.")


if __name__ == "__main__":
    main()
