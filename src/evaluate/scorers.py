"""Per-turn scoring for DiagFlowBench evaluation.

Three Jaccard-based on-procedure capability scorers: position tracking,
branch following, and termination recognition. Off-procedure failure-mode
classification is handled by src.evaluate.judge (LLM-as-judge).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _text_overlap(text_a: str, text_b: str) -> float:
    """Compute word-level Jaccard similarity between two texts."""
    words_a = set(_normalize_text(text_a).split())
    words_b = set(_normalize_text(text_b).split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def raw_best_jaccard(
    model_action: str,
    graph_nodes: dict[str, Any],
) -> tuple[str | None, float]:
    """Return (best_node_id, best_score) with no threshold filtering.

    Used by analysis.py to compute off-graph rate at any threshold τ:
      off_graph = best_score < τ

    Separates fabrication (no node matches) from misnavigation (wrong node
    matches) without committing to a fixed threshold at scoring time.
    """
    best_node = None
    best_score = 0.0
    for nid, node in graph_nodes.items():
        score = _text_overlap(model_action, node.get("text", ""))
        if score > best_score:
            best_score = score
            best_node = nid
    return best_node, round(best_score, 4)


def _find_best_matching_node(
    model_action: str,
    graph_nodes: dict[str, Any],
    *,
    threshold: float,
) -> tuple[str | None, float]:
    """Find the graph node whose text best matches the model's suggestion.

    Returns (node_id, similarity_score) or (None, 0.0) if below threshold.
    """
    best_node, best_score = raw_best_jaccard(model_action, graph_nodes)
    if best_score < threshold:
        return None, 0.0
    return best_node, best_score


def _get_forward_reachable(
    node_id: str,
    adjacency: dict[str, list[tuple[str, str | None]]],
    max_hops: int = 2,
) -> set[str]:
    """Get the set of nodes reachable within max_hops from node_id."""
    reachable = set()
    frontier = {node_id}

    for _ in range(max_hops):
        next_frontier = set()
        for n in frontier:
            for target, _ in adjacency.get(n, []):
                if target not in reachable:
                    next_frontier.add(target)
                    reachable.add(target)
        frontier = next_frontier

    return reachable


def _build_adjacency(graph_data: dict) -> dict[str, list[tuple[str, str | None]]]:
    """Build adjacency list from graph data."""
    adj: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    for edge in graph_data["graph"]["edges"]:
        adj[edge["source"]].append((edge["target"], edge.get("label")))
    return adj


def score_position_tracking(
    model_action: str,
    expected_node_id: str,
    graph_data: dict[str, Any],
    *,
    threshold: float = 0.05,
) -> dict[str, Any]:
    """Score whether the model's action matches the expected current node.

    Also checks whether the matched node is forward-reachable within 2 hops
    (partial credit for misnavigation vs. fabrication).
    """
    nodes = graph_data["graph"]["nodes"]
    adj = _build_adjacency(graph_data)

    matched_node, score = _find_best_matching_node(
        model_action, nodes, threshold=threshold
    )

    if matched_node is None:
        return {
            "correct": False,
            "forward_reachable": False,
            "matched_node": None,
            "expected_node": expected_node_id,
            "similarity": 0.0,
        }

    is_correct = matched_node == expected_node_id
    forward_nodes = _get_forward_reachable(expected_node_id, adj)
    is_forward = matched_node in forward_nodes

    return {
        "correct": is_correct,
        "forward_reachable": is_forward,
        "matched_node": matched_node,
        "expected_node": expected_node_id,
        "similarity": round(score, 4),
    }


def score_branch_following(
    model_action: str,
    decision_node_id: str,
    expected_label: str,
    graph_data: dict[str, Any],
    *,
    threshold: float = 0.05,
) -> dict[str, Any]:
    """Score branch following at decision nodes.

    At turns immediately following a decision node, check which labelled
    successor the model's suggestion aligns with.
    """
    nodes = graph_data["graph"]["nodes"]
    adj = _build_adjacency(graph_data)

    # Get all successors of the decision node with their labels
    successors = adj.get(decision_node_id, [])
    if not successors:
        return {
            "correct_branch": False,
            "predicted_branch": None,
            "expected_branch": expected_label,
            "error": "No successors found for decision node",
        }

    # Match model action against each successor's node text
    best_successor = None
    best_label = None
    best_score = 0.0

    for target, label in successors:
        target_text = nodes.get(target, {}).get("text", "")
        score = _text_overlap(model_action, target_text)
        if score > best_score:
            best_score = score
            best_successor = target
            best_label = label

    if best_score < threshold:
        return {
            "correct_branch": False,
            "predicted_branch": None,
            "expected_branch": expected_label,
            "similarity": 0.0,
        }

    return {
        "correct_branch": (best_label or "") == (expected_label or ""),
        "predicted_branch": best_label,
        "expected_branch": expected_label,
        "matched_successor": best_successor,
        "similarity": round(best_score, 4),
    }


def score_termination(
    model_response: str,
    graph_nodes: dict[str, Any],
    *,
    threshold: float = 0.05,
) -> dict[str, Any]:
    """Score termination recognition at a terminal node.

    TR(t) = 1[rho(a_hat_t) = None]: the response is resolved against every
    graph node via the same Jaccard matching used for step accuracy, at the
    calibrated threshold. The model scores correct when it names no graph
    node (rho resolves to None) and incorrect when it proposes any node,
    since the procedure has already ended.
    """
    matched_node, score = _find_best_matching_node(
        model_response, graph_nodes, threshold=threshold
    )
    return {
        "matched_node": matched_node,
        "similarity": round(score, 4),
        "correct": matched_node is None,
    }


def score_turn(
    *,
    model_response: str,
    turn_index: int,
    expected_node_id: str,
    graph_data: dict[str, Any],
    is_decision_follow: bool = False,
    decision_node_id: str | None = None,
    expected_branch_label: str | None = None,
    is_at_terminator: bool = False,
    threshold: float = 0.05,
) -> dict[str, Any]:
    """Score three primary capabilities for a single model turn.

    Off-graph rate is NOT scored here — it is computed as a diagnostic
    statistic in analysis.py via raw_best_jaccard across the threshold sweep.

    Returns a dict with scores for each applicable capability.
    """
    scores: dict[str, Any] = {"turn_index": turn_index}

    # 1. Position tracking (always scored)
    scores["position_tracking"] = score_position_tracking(
        model_response, expected_node_id, graph_data, threshold=threshold
    )

    # 2. Branch following (scored at decision-follow turns)
    if is_decision_follow and decision_node_id and expected_branch_label is not None:
        scores["branch_following"] = score_branch_following(
            model_response, decision_node_id, expected_branch_label, graph_data,
            threshold=threshold,
        )

    # 3. Termination recognition (scored at terminator turns)
    if is_at_terminator:
        scores["termination"] = score_termination(
            model_response, graph_data["graph"]["nodes"], threshold=threshold
        )

    return scores
