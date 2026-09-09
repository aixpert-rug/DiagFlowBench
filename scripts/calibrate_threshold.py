#!/usr/bin/env python3
"""Threshold calibration for the Jaccard node-matching scorer.

Sweeps candidate thresholds over in-graph (scripted operator turns) and
out-of-graph (injection turns) populations; selects τ* = argmax F1.

Usage:
    python scripts/calibrate_threshold.py
    python scripts/calibrate_threshold.py --conversations-dir path/to/convs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CONVERSATIONS_DIR,
    EVALUATION_DIR,
    SYNTHETIC_GRAPHS_DIR,
    JACCARD_SWEEP_THRESHOLDS,
    JACCARD_THRESHOLD_DEFAULT,
)
from src.evaluate.scorers import raw_best_jaccard


def _load_graph_cache(graph_dir: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    for p in sorted(graph_dir.glob("GRAPH*.json")):
        g = json.loads(p.read_text(encoding="utf-8"))
        cache[g["id"]] = g
    return cache


def _split_graphs(graph_cache: dict[str, dict], val_fraction: float = 0.10) -> tuple[set[str], set[str]]:
    """Deterministic 90/10 split on sorted graph IDs.

    Returns (calibration_ids, validation_ids). The last val_fraction of
    graphs (alphabetically sorted) form the validation set so the split
    is reproducible without a random seed.
    """
    all_ids = sorted(graph_cache.keys())
    n_val = max(1, round(len(all_ids) * val_fraction))
    val_ids = set(all_ids[-n_val:])
    cal_ids = set(all_ids[:-n_val])
    return cal_ids, val_ids


def collect_samples(
    conversations_dir: Path,
    graph_cache: dict[str, dict],
    graph_ids: set[str] | None = None,
) -> tuple[list[float], list[float]]:
    """Return (in_graph_scores, out_of_graph_scores) for the given graph subset.

    graph_ids : restrict to conversations belonging to these graphs.
                If None, all graphs are included.

    in_graph_scores  — best Jaccard for each scripted operator turn in
                       clean (non-mixed) base conversations.
    out_of_graph_scores — best Jaccard for each injection turn in mixed
                          conversations (from injection_log).
    """
    in_graph: list[float] = []
    out_of_graph: list[float] = []

    for f in sorted(conversations_dir.rglob("*.json")):
        if f.stem == "quality_report":
            continue

        conv = json.loads(f.read_text(encoding="utf-8"))
        graph_id = conv.get("graph_id", "")
        if graph_ids is not None and graph_id not in graph_ids:
            continue

        graph_data = graph_cache.get(graph_id)
        if not graph_data:
            continue

        nodes = graph_data["graph"]["nodes"]
        regime = conv.get("regime", "clean")

        if regime == "clean":
            for turn in conv.get("turns", []):
                if turn.get("speaker") != "operator":
                    continue
                if turn.get("node_type") == "terminator":
                    continue  # terminators are short; exclude to avoid bias
                utterance = turn.get("utterance", "")
                if not utterance:
                    continue
                _, score = raw_best_jaccard(utterance, nodes)
                in_graph.append(score)

        elif regime == "mixed":
            log = conv.get("injection_log", [])
            if log:
                for entry in log:
                    utterance = entry.get("utterance", "")
                    if not utterance:
                        continue
                    _, score = raw_best_jaccard(utterance, nodes)
                    out_of_graph.append(score)
            else:
                for turn in conv.get("turns", []):
                    if not turn.get("is_injection"):
                        continue
                    utterance = turn.get("utterance", "")
                    if not utterance:
                        continue
                    _, score = raw_best_jaccard(utterance, nodes)
                    out_of_graph.append(score)

    return in_graph, out_of_graph


def _precision_recall_f1(
    in_scores: list[float],
    out_scores: list[float],
    tau: float,
) -> tuple[float, float, float]:
    """Treat τ as the decision boundary: score ≥ τ → predicted in-graph.

    Positive class = in-graph.
    """
    tp = sum(1 for s in in_scores  if s >= tau)   # correct in-graph
    fp = sum(1 for s in out_scores if s >= tau)   # out-of-graph called in-graph
    fn = sum(1 for s in in_scores  if s < tau)    # in-graph called out-of-graph

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    return precision, recall, f1


def find_optimal_threshold(
    in_scores: list[float],
    out_scores: list[float],
    candidates: list[float],
) -> tuple[float, list[dict]]:
    """Return (τ*, sweep_table) where τ* maximises F1."""
    sweep = []
    for tau in candidates:
        p, r, f1 = _precision_recall_f1(in_scores, out_scores, tau)
        acc = (
            sum(1 for s in in_scores  if s >= tau) +
            sum(1 for s in out_scores if s < tau)
        ) / (len(in_scores) + len(out_scores)) if (in_scores or out_scores) else 0.0
        sweep.append({
            "tau": tau,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "accuracy": round(acc, 4),
        })

    best = max(sweep, key=lambda x: x["f1"])
    return best["tau"], sweep


def _percentiles(scores: list[float]) -> dict:
    if not scores:
        return {}
    s = sorted(scores)
    n = len(s)
    def pct(p: float) -> float:
        return round(s[int(p / 100 * (n - 1))], 4)
    return {
        "n":    n,
        "min":  round(s[0], 4),
        "p10":  pct(10),
        "p25":  pct(25),
        "p50":  pct(50),
        "p75":  pct(75),
        "p90":  pct(90),
        "max":  round(s[-1], 4),
        "mean": round(sum(s) / n, 4),
    }


def run_calibration(
    conversations_dir: Path | None = None,
    graph_dir: Path | None = None,
    output_dir: Path | None = None,
    candidates: list[float] | None = None,
    val_fraction: float = 0.10,
) -> dict:
    if conversations_dir is None:
        conversations_dir = CONVERSATIONS_DIR
    if graph_dir is None:
        graph_dir = SYNTHETIC_GRAPHS_DIR
    if output_dir is None:
        output_dir = EVALUATION_DIR
    if candidates is None:
        candidates = JACCARD_SWEEP_THRESHOLDS

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading graphs and conversations...")
    graph_cache = _load_graph_cache(graph_dir)

    cal_ids, val_ids = _split_graphs(graph_cache, val_fraction)
    print(f"  Graph split: {len(cal_ids)} calibration / {len(val_ids)} validation")
    print(f"  Validation graphs: {sorted(val_ids)}")

    cal_in, cal_out = collect_samples(conversations_dir, graph_cache, graph_ids=cal_ids)
    if not cal_in:
        print("ERROR: No in-graph samples found. Run generation pipeline first.")
        sys.exit(1)
    if not cal_out:
        print("ERROR: No out-of-graph samples found. Run injection pipeline first.")
        sys.exit(1)

    print(f"\n  Calibration set — in-graph: {len(cal_in)}  out-of-graph: {len(cal_out)}")
    tau_star, sweep = find_optimal_threshold(cal_in, cal_out, candidates)

    val_in, val_out = collect_samples(conversations_dir, graph_cache, graph_ids=val_ids)
    val_p, val_r, val_f1 = _precision_recall_f1(val_in, val_out, tau_star)
    val_acc = (
        sum(1 for s in val_in  if s >= tau_star) +
        sum(1 for s in val_out if s <  tau_star)
    ) / (len(val_in) + len(val_out)) if (val_in or val_out) else 0.0

    print(f"  Validation set  — in-graph: {len(val_in)}  out-of-graph: {len(val_out)}")

    report = {
        "tau_star": tau_star,
        "tau_default": JACCARD_THRESHOLD_DEFAULT,
        "tau_default_matches_optimum": tau_star == JACCARD_THRESHOLD_DEFAULT,
        "split": {
            "val_fraction": val_fraction,
            "n_calibration_graphs": len(cal_ids),
            "n_validation_graphs": len(val_ids),
            "calibration_graphs": sorted(cal_ids),
            "validation_graphs": sorted(val_ids),
        },
        "calibration": {
            "populations": {
                "in_graph":     _percentiles(cal_in),
                "out_of_graph": _percentiles(cal_out),
            },
            "sweep": sweep,
        },
        "validation": {
            "tau_star_applied": tau_star,
            "populations": {
                "in_graph":     _percentiles(val_in),
                "out_of_graph": _percentiles(val_out),
            },
            "precision": round(val_p, 4),
            "recall":    round(val_r, 4),
            "f1":        round(val_f1, 4),
            "accuracy":  round(val_acc, 4),
        },
        "note": (
            "tau_star is the Jaccard threshold that maximises F1 on the "
            "calibration split (90% of graphs, sorted deterministically). "
            "Validation metrics are reported on the held-out 10% of graphs "
            "and confirm generalisation of tau_star. "
            "In-graph population: scripted operator turns from base "
            "conversations (verified paraphrases of graph nodes). "
            "Out-of-graph population: injection turns (LLM-generated, "
            "zero-temperature verified to not entail any outgoing edge). "
            "This calibration is reported in the paper appendix."
        ),
    }

    out_path = output_dir / "threshold_calibration.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  Optimal threshold (τ*):  {tau_star}  [fit on {len(cal_ids)} graphs]")
    print(f"  Config default:          {JACCARD_THRESHOLD_DEFAULT}")
    if tau_star != JACCARD_THRESHOLD_DEFAULT:
        print(f"  *** τ* differs from default — update JACCARD_THRESHOLD_DEFAULT "
              f"in config.py to {tau_star} ***")
    print(f"\n  Calibration set:")
    print(f"    In-graph  median Jaccard: {report['calibration']['populations']['in_graph']['p50']}")
    print(f"    Out-graph median Jaccard: {report['calibration']['populations']['out_of_graph']['p50']}")
    print(f"\n  Calibration sweep:")
    print(f"  {'τ':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Acc':>6}")
    for row in sweep:
        marker = " ← τ*" if row["tau"] == tau_star else ""
        print(f"  {row['tau']:>6.2f}  {row['precision']:>6.4f}  "
              f"{row['recall']:>6.4f}  {row['f1']:>6.4f}  "
              f"{row['accuracy']:>6.4f}{marker}")
    print(f"\n  Validation ({len(val_ids)} held-out graphs) at τ*={tau_star}:")
    print(f"    Precision: {val_p:.4f}  Recall: {val_r:.4f}  "
          f"F1: {val_f1:.4f}  Acc: {val_acc:.4f}")
    print(f"\n  Report written to {out_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate Jaccard threshold on in-graph / out-of-graph populations"
    )
    parser.add_argument("--conversations-dir", default=None)
    parser.add_argument("--graph-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_calibration(
        conversations_dir=Path(args.conversations_dir) if args.conversations_dir else None,
        graph_dir=Path(args.graph_dir) if args.graph_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
