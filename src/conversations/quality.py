"""Quality control for generated conversations.

Computes Self-BLEU diversity and edge-label consistency scores across generated conversations.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.config import (
    SELF_BLEU_THRESHOLD,
    EDGE_LABEL_CONSISTENCY_THRESHOLD,
    CONVERSATIONS_DIR,
)


def compute_self_bleu(
    utterances: list[str],
    n: int = 4,
) -> float:
    """Compute Self-BLEU for a set of utterances. Lower = more diverse."""
    if len(utterances) < 2:
        return 0.0

    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    smoother = SmoothingFunction().method1
    scores = []

    for i, hypothesis in enumerate(utterances):
        references = [u for j, u in enumerate(utterances) if j != i]
        hyp_tokens = hypothesis.lower().split()
        ref_tokens = [r.lower().split() for r in references]

        if not hyp_tokens or not any(ref_tokens):
            continue

        try:
            score = sentence_bleu(
                ref_tokens,
                hyp_tokens,
                weights=tuple(1.0 / n for _ in range(n)),
                smoothing_function=smoother,
            )
            scores.append(score)
        except Exception:
            continue

    return sum(scores) / len(scores) if scores else 0.0


def check_edge_label_consistency(
    conversation: dict[str, Any],
) -> dict[str, Any]:
    """Check if each operator utterance is consistent with its edge label."""
    turns = conversation.get("turns", [])
    metadata = conversation.get("metadata", {})
    edge_labels = metadata.get("edge_labels", [])
    decision_points = metadata.get("decision_points", [])

    issues = []
    total_checked = 0
    consistent = 0

    # Build decision point lookup
    dp_labels = {dp["node"]: dp["label"] for dp in decision_points}

    for turn in turns:
        gt = turn.get("ground_truth", {})
        node_type = gt.get("node_type", "")

        if node_type == "decision" and turn.get("speaker") == "operator":
            # This is an operator response to a decision question
            node_id = gt.get("node_id", "")
            expected_label = dp_labels.get(node_id, "")
            utterance = turn.get("utterance", "")

            total_checked += 1
            # Simple keyword check (full LLM-based check done in scoring)
            if expected_label and expected_label.lower() in utterance.lower():
                consistent += 1
            else:
                issues.append({
                    "turn_index": turn.get("turn_index"),
                    "node_id": node_id,
                    "expected_label": expected_label,
                    "utterance_preview": utterance[:100],
                })

    consistency_score = consistent / total_checked if total_checked > 0 else 1.0

    return {
        "total_decision_turns": total_checked,
        "consistent_turns": consistent,
        "consistency_score": round(consistency_score, 4),
        "passed": consistency_score >= EDGE_LABEL_CONSISTENCY_THRESHOLD,
        "issues": issues,
    }


def run_quality_control(
    conversations_dir: Path | None = None,
) -> dict[str, Any]:
    """Run quality control across all generated conversations."""
    if conversations_dir is None:
        conversations_dir = CONVERSATIONS_DIR

    conv_files = sorted(conversations_dir.glob("*.json"))
    if not conv_files:
        return {"error": "No conversation files found"}

    all_operator_utterances: list[str] = []
    per_graph_utterances: dict[str, list[str]] = defaultdict(list)

    consistency_results = []
    total_convs = len(conv_files)

    for cf in conv_files:
        conv = json.loads(cf.read_text(encoding="utf-8"))
        graph_id = conv.get("graph_id", "unknown")

        for turn in conv.get("turns", []):
            if turn.get("speaker") == "operator":
                utt = turn.get("utterance", "")
                if utt:
                    all_operator_utterances.append(utt)
                    per_graph_utterances[graph_id].append(utt)

        consistency = check_edge_label_consistency(conv)
        consistency_results.append(consistency)

    global_self_bleu = compute_self_bleu(all_operator_utterances)
    per_graph_bleu = {}
    for gid, utts in per_graph_utterances.items():
        per_graph_bleu[gid] = round(compute_self_bleu(utts), 4)

    passed_consistency = sum(1 for r in consistency_results if r["passed"])
    avg_consistency = (
        sum(r["consistency_score"] for r in consistency_results) / len(consistency_results)
        if consistency_results else 0.0
    )

    report = {
        "total_conversations": total_convs,
        "total_operator_utterances": len(all_operator_utterances),
        "self_bleu": {
            "global": round(global_self_bleu, 4),
            "threshold": SELF_BLEU_THRESHOLD,
            "passed": global_self_bleu < SELF_BLEU_THRESHOLD,
            "per_graph": per_graph_bleu,
        },
        "edge_label_consistency": {
            "passed_conversations": passed_consistency,
            "total_conversations": total_convs,
            "pass_rate": round(passed_consistency / total_convs, 4) if total_convs else 0,
            "average_score": round(avg_consistency, 4),
            "threshold": EDGE_LABEL_CONSISTENCY_THRESHOLD,
        },
    }

    report_path = conversations_dir / "quality_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report
