"""Root-to-terminator path enumeration for diagnostic flowcharts.

DFS enumeration with loop bounding (MAX_LOOP_ITERATIONS). Stratified selection
ensures every decision-node outgoing edge is exercised at least once. Outputs a
path manifest with full metadata for downstream conversation generation.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from src.config import (
    SYNTHETIC_GRAPHS_DIR,
    PATHS_DIR,
    MAX_LOOP_ITERATIONS,
    ROBUSTNESS_VARIANTS,
    GRAPH_SIZE_SMALL_MAX,
    GRAPH_SIZE_MEDIUM_MAX,
)


@dataclass
class DecisionPoint:
    """A decision point encountered on a path."""
    node: str
    label: str
    question: str


@dataclass
class PathRecord:
    """A single root-to-terminator path through a graph."""
    path_id: str
    graph_id: str
    node_sequence: list[str]
    edge_labels: list[str | None]
    decision_points: list[DecisionPoint]
    terminator_type: str
    terminator_text: str
    depth: int
    num_decisions: int
    num_processes: int
    num_documents: int
    true_terminator: bool  # False when path ends at a dead-end process/decision node

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class GraphStats:
    """Per-graph statistics."""
    graph_id: str
    description: str
    num_nodes: int
    num_edges: int
    num_decisions: int
    num_processes: int
    num_terminators: int
    num_documents: int
    num_paths: int
    num_conversations: int
    branch_coverage: float
    size_category: str
    terminator_types: list[str]
    has_loops: bool


def load_graph(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_all_graphs() -> list[dict[str, Any]]:
    """Load all graphs from the synthetic directory, sorted by ID."""
    graphs = []
    for p in sorted(SYNTHETIC_GRAPHS_DIR.glob("GRAPH*.json")):
        graphs.append(load_graph(p))
    return graphs


def _build_adjacency(graph_data: dict) -> dict[str, list[tuple[str, str | None]]]:
    """Build adjacency list: node → [(target, label), ...]."""
    adj: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    for edge in graph_data["graph"]["edges"]:
        src = edge["source"]
        tgt = edge["target"]
        label = edge.get("label")
        adj[src].append((tgt, label))
    return adj


def _find_start_nodes(graph_data: dict) -> list[str]:
    """Find all start nodes (terminator type whose text begins with 'Start')."""
    starts = []
    for nid, node in graph_data["graph"]["nodes"].items():
        if node["type"] == "terminator" and node["text"].strip().lower().startswith("start"):
            starts.append(nid)
    return starts


def _find_terminator_nodes(graph_data: dict) -> set[str]:
    """Find all terminator nodes that are not start nodes."""
    terminators = set()
    for nid, node in graph_data["graph"]["nodes"].items():
        if node["type"] == "terminator" and not node["text"].strip().lower().startswith("start"):
            terminators.add(nid)
    return terminators


def _classify_terminator(text: str) -> str:
    """Classify terminator node into one of three types (§3)."""
    text_lower = text.lower()
    if "escalate" in text_lower:
        return "escalate"
    elif "hold" in text_lower or "stop" in text_lower:
        return "production_hold"
    else:
        return "procedure_complete"


def _has_loops(graph_data: dict) -> bool:
    """Detect if the graph has cycles (retry loops)."""
    adj = _build_adjacency(graph_data)
    nodes = set(graph_data["graph"]["nodes"].keys())

    # Simple DFS cycle detection
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in nodes}

    def dfs(u: str) -> bool:
        color[u] = GRAY
        for v, _ in adj.get(u, []):
            if color.get(v, WHITE) == GRAY:
                return True
            if color.get(v, WHITE) == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False

    for n in nodes:
        if color[n] == WHITE:
            if dfs(n):
                return True
    return False


def _classify_size(num_nodes: int) -> str:
    if num_nodes <= GRAPH_SIZE_SMALL_MAX:
        return "small"
    elif num_nodes <= GRAPH_SIZE_MEDIUM_MAX:
        return "medium"
    else:
        return "large"


# Path enumeration (DFS with loop bounding)
def enumerate_paths(
    graph_data: dict,
    max_loop_iters: int = MAX_LOOP_ITERATIONS,
) -> list[list[tuple[str, str | None]]]:
    """Enumerate all root-to-terminator paths in a graph.

    Each path is a list of (node_id, edge_label_used_to_reach_it) tuples.
    Loops are bounded: each node can be visited at most (1 + max_loop_iters) times.
    """
    adj = _build_adjacency(graph_data)
    starts = _find_start_nodes(graph_data)
    terminators = _find_terminator_nodes(graph_data)
    nodes = graph_data["graph"]["nodes"]

    if not starts:
        return []

    all_paths: list[list[tuple[str, str | None]]] = []

    for start in starts:
        # DFS with visit counting for loop bounding
        stack: list[tuple[str, str | None, list[tuple[str, str | None]], dict[str, int]]] = [
            (start, None, [], defaultdict(int))
        ]

        while stack:
            node, label, path_so_far, visit_counts = stack.pop()

            # Check loop bound
            new_counts = defaultdict(int, visit_counts)
            new_counts[node] += 1
            if new_counts[node] > 1 + max_loop_iters:
                continue  # Exceeded loop bound

            current_path = path_so_far + [(node, label)]

            # Check if terminator reached
            if node in terminators:
                all_paths.append(current_path)
                continue

            # Expand successors
            successors = adj.get(node, [])
            if not successors:
                # Dead-end node (no outgoing edges). Many decision branches
                # end at process nodes without explicit terminator nodes.
                # Treat these as implicit procedure endpoints so that all
                # branches are reachable.
                all_paths.append(current_path)
                continue

            for target, edge_label in successors:
                stack.append((target, edge_label, current_path, new_counts))

    return all_paths


def _compute_branch_coverage(
    graph_data: dict,
    paths: list[list[tuple[str, str | None]]],
) -> float:
    """Compute fraction of decision-node outgoing edges covered by paths."""
    nodes = graph_data["graph"]["nodes"]
    adj = _build_adjacency(graph_data)

    # Collect all decision-node edges
    decision_edges: set[tuple[str, str]] = set()
    for nid, node in nodes.items():
        if node["type"] == "decision":
            for target, label in adj.get(nid, []):
                edge_key = (nid, label or "")
                decision_edges.add(edge_key)

    if not decision_edges:
        return 1.0

    # Collect covered edges
    covered: set[tuple[str, str]] = set()
    for path in paths:
        for i, (node, label) in enumerate(path):
            if nodes.get(node, {}).get("type") == "decision" and i + 1 < len(path):
                next_node, next_label = path[i + 1]
                # The label on the edge is the label used to reach the NEXT node
                covered.add((node, next_label or ""))

    return len(covered & decision_edges) / len(decision_edges)


def _edges_covered_by_path(
    path: list[tuple[str, str | None]],
    nodes: dict[str, Any],
) -> set[tuple[str, str]]:
    """Return the set of (decision_node, label) edges covered by a path."""
    covered: set[tuple[str, str]] = set()
    for i, (node, label) in enumerate(path):
        if nodes.get(node, {}).get("type") == "decision" and i + 1 < len(path):
            next_label = path[i + 1][1]
            covered.add((node, next_label or ""))
    return covered


def _terminator_of_path(
    path: list[tuple[str, str | None]],
    nodes: dict[str, Any],
) -> str:
    """Return the terminator type of a path's final node."""
    last_node = path[-1][0]
    return _classify_terminator(nodes.get(last_node, {}).get("text", ""))


def select_stratified_paths(
    graph_data: dict,
    all_paths: list[list[tuple[str, str | None]]],
    *,
    min_paths_per_graph: int = 2,
    max_paths_per_graph: int = 20,
) -> list[list[tuple[str, str | None]]]:
    """Select a stratified subset of paths ensuring full branch coverage.

    Strategy:
    1. Greedy set-cover: pick paths until every decision-node outgoing edge
       is exercised at least once.
    2. Ensure every terminator type is represented at least once.
    3. Add diversity paths up to a cap proportional to the graph's decision count.
    """
    if not all_paths:
        return []

    nodes = graph_data["graph"]["nodes"]
    adj = _build_adjacency(graph_data)

    # All decision edges in the graph
    all_decision_edges: set[tuple[str, str]] = set()
    for nid, node in nodes.items():
        if node["type"] == "decision":
            for target, label in adj.get(nid, []):
                all_decision_edges.add((nid, label or ""))

    path_edges = [_edges_covered_by_path(p, nodes) for p in all_paths]
    path_terminators = [_terminator_of_path(p, nodes) for p in all_paths]

    selected_indices: list[int] = []
    covered_edges: set[tuple[str, str]] = set()
    covered_terminators: set[str] = set()

    # Phase 1: Greedy set-cover for decision edges
    uncovered = set(all_decision_edges)
    used = set()

    while uncovered:
        # Pick path that covers the most uncovered edges
        best_idx = -1
        best_new = 0
        for i, edges in enumerate(path_edges):
            if i in used:
                continue
            new_coverage = len(edges & uncovered)
            if new_coverage > best_new:
                best_new = new_coverage
                best_idx = i

        if best_idx == -1 or best_new == 0:
            break  # No path can cover remaining edges (disconnected graph)

        selected_indices.append(best_idx)
        used.add(best_idx)
        covered_edges |= path_edges[best_idx]
        covered_terminators.add(path_terminators[best_idx])
        uncovered -= path_edges[best_idx]

    # Phase 2: Ensure all terminator types are covered
    all_terminator_types = set(path_terminators)
    for ttype in all_terminator_types:
        if ttype not in covered_terminators:
            for i, pt in enumerate(path_terminators):
                if pt == ttype and i not in used:
                    selected_indices.append(i)
                    used.add(i)
                    covered_terminators.add(ttype)
                    covered_edges |= path_edges[i]
                    break

    # Phase 3: Add diversity paths up to cap
    # Cap is proportional to number of decision nodes (more branching = more paths)
    num_decisions = len(set(nid for nid, _ in all_decision_edges))
    diversity_cap = max(min_paths_per_graph, min(max_paths_per_graph, num_decisions + 2))

    # Add paths sorted by depth diversity (prefer different lengths)
    remaining = [i for i in range(len(all_paths)) if i not in used]
    # Sort by path length to get diversity
    remaining.sort(key=lambda i: len(all_paths[i]))

    # Sample evenly across the length distribution
    if remaining and len(selected_indices) < diversity_cap:
        needed = diversity_cap - len(selected_indices)
        step = max(1, len(remaining) // needed)
        for j in range(0, len(remaining), step):
            if len(selected_indices) >= diversity_cap:
                break
            idx = remaining[j]
            selected_indices.append(idx)

    # Ensure minimum
    if len(selected_indices) < min_paths_per_graph:
        for i in range(len(all_paths)):
            if i not in set(selected_indices) and len(selected_indices) < min_paths_per_graph:
                selected_indices.append(i)

    # De-duplicate and sort by index for determinism
    selected_indices = sorted(set(selected_indices))

    return [all_paths[i] for i in selected_indices]


def path_to_record(
    graph_data: dict,
    path: list[tuple[str, str | None]],
    path_idx: int,
) -> PathRecord:
    """Convert a raw path into a PathRecord with metadata."""
    graph_id = graph_data["id"]
    nodes = graph_data["graph"]["nodes"]

    node_sequence = [node for node, _ in path]
    edge_labels = [label for _, label in path]

    decision_points = []
    for i, (node, label) in enumerate(path):
        node_info = nodes.get(node, {})
        if node_info.get("type") == "decision" and i + 1 < len(path):
            next_label = path[i + 1][1]
            decision_points.append(DecisionPoint(
                node=node,
                label=next_label or "",
                question=node_info.get("text", ""),
            ))

    last_node = node_sequence[-1]
    last_info = nodes.get(last_node, {})
    terminator_text = last_info.get("text", "")
    terminator_type = _classify_terminator(terminator_text)

    num_decisions = sum(1 for n in node_sequence if nodes.get(n, {}).get("type") == "decision")
    num_processes = sum(1 for n in node_sequence if nodes.get(n, {}).get("type") == "process")
    num_documents = sum(1 for n in node_sequence if nodes.get(n, {}).get("type") == "document")

    return PathRecord(
        path_id=f"{graph_id}_P{path_idx:03d}",
        graph_id=graph_id,
        node_sequence=node_sequence,
        edge_labels=edge_labels,
        decision_points=decision_points,
        terminator_type=terminator_type,
        terminator_text=terminator_text,
        depth=len(node_sequence),
        num_decisions=num_decisions,
        num_processes=num_processes,
        num_documents=num_documents,
        true_terminator=last_info.get("type") == "terminator",
    )


def enumerate_all(
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Enumerate paths across all 50 graphs and produce a manifest.

    Returns the manifest dict and writes it to disk.
    """
    if output_dir is None:
        output_dir = PATHS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    graphs = load_all_graphs()
    all_path_records: list[dict] = []
    per_graph: dict[str, dict] = {}
    global_decision_edges = 0
    global_covered_edges = 0

    total_stats = {
        "total_graphs": len(graphs),
        "total_nodes": 0,
        "total_edges": 0,
        "total_decisions": 0,
        "total_processes": 0,
        "total_terminators": 0,
        "total_documents": 0,
        "size_distribution": {"small": 0, "medium": 0, "large": 0},
    }

    for graph_data in graphs:
        gid = graph_data["id"]
        nodes = graph_data["graph"]["nodes"]
        edges = graph_data["graph"]["edges"]

        type_counts: dict[str, int] = defaultdict(int)
        for nid, node in nodes.items():
            type_counts[node["type"]] += 1

        num_nodes = len(nodes)
        num_edges = len(edges)
        size_cat = _classify_size(num_nodes)
        has_loops = _has_loops(graph_data)

        raw_paths = enumerate_paths(graph_data)
        exhaustive_count = len(raw_paths)

        selected_paths = select_stratified_paths(graph_data, raw_paths)
        branch_cov = _compute_branch_coverage(graph_data, selected_paths)

        records = []
        for idx, path in enumerate(selected_paths):
            record = path_to_record(graph_data, path, idx)
            records.append(record)
            all_path_records.append(record.to_dict())

        terminator_types = list(set(r.terminator_type for r in records))

        num_conversations = len(records) * (1 + ROBUSTNESS_VARIANTS)  # base + variants

        graph_stats = GraphStats(
            graph_id=gid,
            description=graph_data.get("description", ""),
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_decisions=type_counts.get("decision", 0),
            num_processes=type_counts.get("process", 0),
            num_terminators=type_counts.get("terminator", 0),
            num_documents=type_counts.get("document", 0),
            num_paths=len(records),
            num_conversations=num_conversations,
            branch_coverage=round(branch_cov, 4),
            size_category=size_cat,
            terminator_types=terminator_types,
            has_loops=has_loops,
        )

        per_graph[gid] = asdict(graph_stats)

        total_stats["total_nodes"] += num_nodes
        total_stats["total_edges"] += num_edges
        total_stats["total_decisions"] += type_counts.get("decision", 0)
        total_stats["total_processes"] += type_counts.get("process", 0)
        total_stats["total_terminators"] += type_counts.get("terminator", 0)
        total_stats["total_documents"] += type_counts.get("document", 0)
        total_stats["size_distribution"][size_cat] += 1

    total_paths = len(all_path_records)
    total_conversations = total_paths * (1 + ROBUSTNESS_VARIANTS)  # base + variants per path

    manifest = {
        "total_paths": total_paths,
        "total_conversations": total_conversations,
        "robustness_multiplier": ROBUSTNESS_VARIANTS,
        "max_loop_iterations": MAX_LOOP_ITERATIONS,
        "graph_statistics": total_stats,
        "per_graph": per_graph,
        "paths": all_path_records,
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_path = output_dir / "summary.txt"
    summary_lines = [
        "DIAGFLOWBENCH — Path Enumeration Summary",
        "",
        f"Total graphs:          {total_stats['total_graphs']}",
        f"Total nodes:           {total_stats['total_nodes']}",
        f"Total edges:           {total_stats['total_edges']}",
        f"Total decision nodes:  {total_stats['total_decisions']}",
        f"Total process nodes:   {total_stats['total_processes']}",
        f"Total terminator nodes:{total_stats['total_terminators']}",
        f"Total document nodes:  {total_stats['total_documents']}",
        "",
        f"Size distribution:",
        f"  Small  (≤{GRAPH_SIZE_SMALL_MAX} nodes):  {total_stats['size_distribution']['small']} graphs",
        f"  Medium ({GRAPH_SIZE_SMALL_MAX+1}–{GRAPH_SIZE_MEDIUM_MAX} nodes): {total_stats['size_distribution']['medium']} graphs",
        f"  Large  (>{GRAPH_SIZE_MEDIUM_MAX} nodes):  {total_stats['size_distribution']['large']} graphs",
        "",
        f"Total root-to-terminator paths: {total_paths}",
        f"Robustness multiplier:          ×{ROBUSTNESS_VARIANTS}",
        f"Total conversations:            {total_conversations}",
        f"Max loop iterations:            {MAX_LOOP_ITERATIONS}",
        "",
        "Per-graph breakdown:",
        "-" * 80,
        f"{'Graph':<10} {'Nodes':>6} {'Edges':>6} {'Dec':>4} {'Size':<8} {'Paths':>6} {'Convs':>6} {'BrCov':>6} {'Loops':<6}",
        "-" * 80,
    ]

    for gid in sorted(per_graph.keys()):
        gs = per_graph[gid]
        summary_lines.append(
            f"{gs['graph_id']:<10} {gs['num_nodes']:>6} {gs['num_edges']:>6} "
            f"{gs['num_decisions']:>4} {gs['size_category']:<8} {gs['num_paths']:>6} "
            f"{gs['num_conversations']:>6} {gs['branch_coverage']:>6.2f} {'yes' if gs['has_loops'] else 'no':<6}"
        )

    summary_lines.extend([
        "-" * 80,
        f"{'TOTAL':<10} {total_stats['total_nodes']:>6} {total_stats['total_edges']:>6} "
        f"{total_stats['total_decisions']:>4} {'':>8} {total_paths:>6} {total_conversations:>6}",
    ])

    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    return manifest
