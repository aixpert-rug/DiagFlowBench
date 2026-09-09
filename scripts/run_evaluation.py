#!/usr/bin/env python3
"""Run the DiagFlowBench LLM evaluation.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --models openai/gpt-4.1,anthropic/claude-sonnet-4-6
    python scripts/run_evaluation.py --max-conversations 10
    python scripts/run_evaluation.py --workers 4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EVAL_MODELS, EVALUATION_DIR
from src.evaluate.runner import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM evaluation for DIAGFLOWBENCH"
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated OpenRouter model IDs to evaluate (default: all 12 from config)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between API calls (default: 0.5)",
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        help="Limit number of conversations to evaluate (for testing)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel model threads (default: one per model)",
    )
    args = parser.parse_args()

    if args.models:
        model_ids = [m.strip() for m in args.models.split(",")]
        models_to_run = []
        for mid in model_ids:
            matched = next((m for m in EVAL_MODELS if m["id"] == mid), None)
            if matched:
                models_to_run.append(matched)
            else:
                models_to_run.append({"name": mid.split("/")[-1], "id": mid, "tier": "custom"})
    else:
        models_to_run = EVAL_MODELS

    print("DiagFlowBench — LLM Evaluation")
    print(f"Models:            {len(models_to_run)}")
    for m in models_to_run:
        print(f"  - {m['name']} ({m['id']})")
    print(f"Output directory:  {EVALUATION_DIR}")
    print(f"Delay:             {args.delay}s")
    if args.max_conversations:
        print(f"Max conversations: {args.max_conversations}")
    print()

    results = run_evaluation(
        models=models_to_run,
        delay=args.delay,
        max_conversations=args.max_conversations,
        max_workers=args.workers,
    )

    print("\nEvaluation complete.")
    print(f"Results saved to {EVALUATION_DIR}")


if __name__ == "__main__":
    main()
