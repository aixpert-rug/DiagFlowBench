#!/usr/bin/env python3
"""Run path enumeration across all 50 diagnostic flowcharts.

Produces:
  - data/output/paths/manifest.json  (machine-readable, full path details)
  - data/output/paths/summary.txt    (human-readable summary)

Usage:
    python scripts/run_path_enumeration.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths.enumerator import enumerate_all


def main() -> None:
    print("DIAGFLOWBENCH — Path Enumeration")

    manifest = enumerate_all()

    total_paths = manifest["total_paths"]
    total_convs = manifest["total_conversations"]
    stats = manifest["graph_statistics"]

    print(f"\nGraphs processed:       {stats['total_graphs']}")
    print(f"Total nodes:            {stats['total_nodes']}")
    print(f"Total edges:            {stats['total_edges']}")
    print(f"Total decision nodes:   {stats['total_decisions']}")
    print()
    print(f"Size distribution:")
    for cat, count in stats["size_distribution"].items():
        print(f"  {cat}: {count}")
    print()
    print(f"Root-to-terminator paths: {total_paths}")
    variant_word = "variant" if manifest['robustness_multiplier'] == 1 else "variants"
    print(f"× {manifest['robustness_multiplier']} robustness {variant_word}")
    print(f"= {total_convs} total conversations")
    print()
    print(f"Manifest written to: data/output/paths/manifest.json")
    print(f"Summary written to:  data/output/paths/summary.txt")


if __name__ == "__main__":
    main()
