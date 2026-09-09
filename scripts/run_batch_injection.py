#!/usr/bin/env python3
"""Generate mixed conversations using the Anthropic Batch API (injection pipeline).

Three phases, each saves its batch ID to batch_injection_state.json.

Usage:
    python scripts/run_batch_injection.py --phase submit-gen
    python scripts/run_batch_injection.py --phase collect-gen
    python scripts/run_batch_injection.py --phase collect-verify
    python scripts/run_batch_injection.py --phase collect-gen --batch-id msgbatch_xxx
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONVERSATIONS_DIR, SYNTHETIC_GRAPHS_DIR
from src.conversations.batch_injection import (
    collect_gen_and_submit_verification,
    collect_verify_and_assemble,
    submit_injection_generation_batch,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DIAGFLOWBENCH — Batch API injection pipeline (Phase 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=["submit-gen", "collect-gen", "collect-verify"],
        help=(
            "submit-gen: plan injections and submit generation batch | "
            "collect-gen: poll generation batch, submit verification batch | "
            "collect-verify: poll verification batch, assemble mixed conversations"
        ),
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Anthropic batch ID (optional; loaded from batch_injection_state.json if omitted)",
    )
    parser.add_argument(
        "--conversations-dir",
        default=str(CONVERSATIONS_DIR),
        help="Directory containing clean conversation JSONs (default: DiagFlowBench_Dataset/conversations)",
    )
    parser.add_argument(
        "--graph-dir",
        default=str(SYNTHETIC_GRAPHS_DIR),
        help="Directory containing graph JSON files",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for mixed conversations (default: same as --conversations-dir)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for injection position and category sampling (default: 42)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Regenerate mixed conversations even if output files already exist",
    )
    args = parser.parse_args()

    conversations_dir = Path(args.conversations_dir)
    graph_dir = Path(args.graph_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.phase == "submit-gen":
        batch_id = submit_injection_generation_batch(
            conversations_dir=conversations_dir,
            graph_dir=graph_dir,
            seed=args.seed,
        )
        print(f"\nPhase 1 complete. Generation batch ID: {batch_id}")
        print("Run again with --phase collect-gen when the batch has processed.")

    elif args.phase == "collect-gen":
        batch_id = collect_gen_and_submit_verification(batch_id=args.batch_id)
        print(f"\nPhase 2 complete. Verification batch ID: {batch_id}")
        print("Run again with --phase collect-verify when the batch has processed.")

    elif args.phase == "collect-verify":
        output_files = collect_verify_and_assemble(
            batch_id=args.batch_id,
            output_dir=output_dir,
            skip_existing=not args.no_skip,
        )
        print(f"\nDone. {len(output_files)} mixed conversations written.")
        print("Next step: python scripts/calibrate_threshold.py")


if __name__ == "__main__":
    main()
