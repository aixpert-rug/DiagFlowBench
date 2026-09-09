"""Compute inter-annotator agreement between human labels and LLM judge.

Reads the CSV produced by sample_iaa.py (with human_label filled in) and
reports:
  - N labelled turns
  - Raw agreement rate
  - Cohen's κ (weighted and unweighted)
  - Per-class precision / recall / F1 (judge treated as classifier, human as gold)
  - Confusion matrix
  - Breakdown by model and injection category

Usage:
    python scripts/compute_iaa.py [--in iaa_sample.csv]

Requires: scikit-learn (pip install scikit-learn)
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

try:
    from sklearn.metrics import (
        cohen_kappa_score,
        classification_report,
        confusion_matrix,
    )
except ImportError:
    print("scikit-learn not found. Install with: pip install scikit-learn")
    sys.exit(1)


VALID_LABELS = {"correct_abstention", "forced_mapping", "fabrication"}
LABEL_ORDER  = ["correct_abstention", "forced_mapping", "fabrication"]


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _validate(rows: list[dict]) -> list[dict]:
    labelled: list[dict] = []
    skipped = 0
    bad_label_rows: list[int] = []

    for row in rows:
        hl = row.get("human_label", "").strip().lower()
        if not hl:
            skipped += 1
            continue
        if hl not in VALID_LABELS:
            bad_label_rows.append(int(row.get("sample_id", 0)))
            continue
        row["human_label"] = hl
        row["judge_label"]  = row.get("judge_label", "").strip().lower()
        labelled.append(row)

    if skipped:
        print(f"Skipped {skipped} rows with empty human_label")
    if bad_label_rows:
        print(f"Warning: unrecognised human_label in sample_id {bad_label_rows}")
        print(f"  Valid values: {sorted(VALID_LABELS)}")

    return labelled


def _confusion(human: list[str], judge: list[str]) -> None:
    cm = confusion_matrix(human, judge, labels=LABEL_ORDER)
    col_w = max(len(l) for l in LABEL_ORDER) + 2
    header = f"{'':>{col_w}}" + "".join(f"{l:>{col_w}}" for l in LABEL_ORDER)
    print(header)
    for i, label in enumerate(LABEL_ORDER):
        row_str = f"{label:>{col_w}}" + "".join(f"{cm[i][j]:>{col_w}}" for j in range(len(LABEL_ORDER)))
        print(row_str)


def _breakdown(labelled: list[dict], group_key: str) -> None:
    by_group: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in labelled:
        g  = row[group_key]
        h  = row["human_label"]
        j  = row["judge_label"]
        by_group[g]["total"] += 1
        if h == j:
            by_group[g]["agree"] += 1

    for g in sorted(by_group.keys()):
        d = by_group[g]
        rate = d["agree"] / d["total"]
        print(f"  {g}: {d['agree']}/{d['total']} ({rate:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute IAA between human and judge labels")
    parser.add_argument("--in", dest="input", default="iaa_sample.csv", help="Filled CSV file")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"File not found: {in_path}")
        sys.exit(1)

    rows     = _load_csv(in_path)
    labelled = _validate(rows)

    if not labelled:
        print("No labelled rows found. Fill in the human_label column first.")
        sys.exit(1)

    human = [r["human_label"] for r in labelled]
    judge = [r["judge_label"]  for r in labelled]

    n       = len(labelled)
    n_agree = sum(h == j for h, j in zip(human, judge))
    acc     = n_agree / n
    kappa   = cohen_kappa_score(human, judge)

    print(f"\nIAA Results — N={n}")
    print(f"  Raw agreement: {n_agree}/{n} ({acc:.1%})")
    print(f"  Cohen's κ:     {kappa:.3f}")
    print()

    print("Per-class report (judge vs human-gold):")
    print(classification_report(
        human, judge,
        labels=LABEL_ORDER,
        target_names=LABEL_ORDER,
        zero_division=0,
    ))

    print("Confusion matrix (rows=human, cols=judge):")
    _confusion(human, judge)
    print()

    print("Agreement by model:")
    _breakdown(labelled, "model")
    print()

    print("Agreement by injection category:")
    _breakdown(labelled, "injection_category")
    print()

    disagreements = [r for r in labelled if r["human_label"] != r["judge_label"]]
    print(f"Disagreements: {len(disagreements)} ({len(disagreements)/n:.1%})")
    if disagreements:
        print("\nDisagreement cases:")
        for r in disagreements:
            print(f"  [{r['sample_id']}] {r['model']} | {r['injection_category']}")
            print(f"    Judge: {r['judge_label']}  |  Human: {r['human_label']}")
            print(f"    Utterance: {r['operator_utterance'][:120]}")
            print(f"    Response:  {r['model_response'][:120]}")
            if r.get("notes"):
                print(f"    Notes: {r['notes']}")
            print()


if __name__ == "__main__":
    main()
