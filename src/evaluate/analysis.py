"""Evaluation analysis for DiagFlowBench.

Computes on-procedure metrics (branch following, termination, position tracking)
for clean conversations and off-procedure failure-mode rates (fabrication,
forced-mapping, correct-abstention, inappropriate-abstention) for mixed
conversations. Also provides density-effect and dynamics analyses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import (
    EVALUATION_DIR,
    MODEL_OUTPUTS_DIR,
    SYNTHETIC_GRAPHS_DIR,
    JACCARD_SWEEP_THRESHOLDS,
    JACCARD_THRESHOLD_DEFAULT,
)
from src.evaluate.judge import (
    classify_injection_turn,
    classify_post_injection_turn,
    _load_cache,
    _save_cache,
)
from src.evaluate.scorers import (
    _build_adjacency,
    _get_forward_reachable,
    _text_overlap,
    raw_best_jaccard,
    score_branch_following,
    score_position_tracking,
    score_termination,
)


def _load_graph_cache(graph_dir: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    for p in sorted(graph_dir.glob("GRAPH*.json")):
        g = json.loads(p.read_text(encoding="utf-8"))
        cache[g["id"]] = g
    return cache


def _empty_clean_acc(thresholds: list[float]) -> dict[float, dict]:
    return {
        tau: {
            "pt_exact": 0, "pt_forward": 0,
            "bf_correct": 0, "bf_total": 0,
            "term_correct": 0, "term_total": 0,
            "off_graph": 0, "total_turns": 0,
        }
        for tau in thresholds
    }


def _empty_mixed_acc() -> dict:
    """Accumulators for mixed-conversation failure-mode analysis.

    Injection-turn classification is produced by the Haiku judge (src.evaluate.judge).
    Jaccard is NOT used to classify injection turns; it is used only for
    on-procedure metrics (branch following, position tracking).
    """
    return {
        "injection_total": 0,
        "fabrication": 0,
        "forced_mapping": 0,
        "correct_abstention": 0,
        "by_category": {
            "coverage_gap":             {"total": 0, "fabrication": 0, "forced_mapping": 0, "correct_abstention": 0, "post_total": 0, "inappropriate_abstention": 0},
            "undocumented_malfunction": {"total": 0, "fabrication": 0, "forced_mapping": 0, "correct_abstention": 0, "post_total": 0, "inappropriate_abstention": 0},
            "unrelated_question":       {"total": 0, "fabrication": 0, "forced_mapping": 0, "correct_abstention": 0, "post_total": 0, "inappropriate_abstention": 0},
        },
        "abstention_with_redirect": 0,
        "forced_mapping_nearby": 0,
        "recovery_total": 0,
        "recovery_correct": 0,
        "post_injection_total": 0,
        "inappropriate_abstention": 0,
        "pre_pt_exact": 0, "pre_total": 0,
        "post_pt_exact": 0, "post_total": 0,
    }


def _clean_rates(acc: dict) -> dict[str, float | None]:
    n = acc["total_turns"]
    return {
        "position_tracking_exact":   acc["pt_exact"]    / n                if n else 0.0,
        "position_tracking_forward": acc["pt_forward"]  / n                if n else 0.0,
        "branch_following":          acc["bf_correct"]  / acc["bf_total"]  if acc["bf_total"]  else None,
        "termination_recognition":   acc["term_correct"] / acc["term_total"] if acc["term_total"] else None,
        "off_graph_rate":            acc["off_graph"]   / n                if n else 0.0,
    }


def _mixed_rates(acc: dict) -> dict[str, Any]:
    inj    = acc["injection_total"]
    post   = acc["post_injection_total"]
    pre_n  = acc["pre_total"]
    post_n = acc["post_total"]
    ca     = acc["correct_abstention"]
    fm     = acc["forced_mapping"]
    rt     = acc["recovery_total"]

    by_cat = {}
    for cat, c in acc["by_category"].items():
        t = c["total"]
        pt = c["post_total"]
        by_cat[cat] = {
            "fabrication":              c["fabrication"]            / t  if t  else None,
            "forced_mapping":           c["forced_mapping"]         / t  if t  else None,
            "correct_abstention":       c["correct_abstention"]     / t  if t  else None,
            "inappropriate_abstention": c["inappropriate_abstention"] / pt if pt else None,
        }

    return {
        "fabrication_rate":          acc["fabrication"]        / inj  if inj  else None,
        "forced_mapping_rate":       acc["forced_mapping"]     / inj  if inj  else None,
        "correct_abstention_rate":   acc["correct_abstention"] / inj  if inj  else None,
        "abstention_redirect_rate":  acc["abstention_with_redirect"] / ca if ca else None,
        "forced_mapping_nearby_rate": acc["forced_mapping_nearby"] / fm if fm else None,
        "recovery_rate":             acc["recovery_correct"] / rt if rt else None,
        "inappropriate_abstention_rate": acc["inappropriate_abstention"] / post if post else None,
        "contamination": {
            "pre_injection_pt_exact":  acc["pre_pt_exact"]  / pre_n  if pre_n  else None,
            "post_injection_pt_exact": acc["post_pt_exact"] / post_n if post_n else None,
            "delta": (acc["post_pt_exact"] / post_n - acc["pre_pt_exact"] / pre_n)
                     if (pre_n and post_n) else None,
        },
        "by_injection_category": by_cat,
    }


def _score_clean_conversation(
    conv_result: dict[str, Any],
    graph_data: dict[str, Any],
    thresholds: list[float],
) -> dict[float, dict]:
    nodes = graph_data["graph"]["nodes"]
    acc = _empty_clean_acc(thresholds)
    score_termination_this_conv: bool = conv_result.get("metadata", {}).get("true_terminator", True)

    for turn in conv_result.get("per_turn", []):
        if turn.get("is_injection"):
            continue

        response      = turn.get("model_response", "")
        gt            = turn.get("ground_truth", {})
        expected_node = gt.get("node_id", "")
        node_type     = gt.get("node_type", "")
        edge_label    = turn.get("edge_label_to_next")

        is_decision   = node_type == "decision"
        is_terminator = node_type == "terminator" and score_termination_this_conv

        _, best_jaccard = raw_best_jaccard(response, nodes)

        for tau in thresholds:
            a = acc[tau]
            a["total_turns"] += 1

            if best_jaccard < tau:
                a["off_graph"] += 1

            pt = score_position_tracking(response, expected_node, graph_data, threshold=tau)
            if pt["correct"]:
                a["pt_exact"] += 1
            if pt["correct"] or pt.get("forward_reachable"):
                a["pt_forward"] += 1

            if is_decision and edge_label is not None:
                bf = score_branch_following(
                    response, expected_node, edge_label, graph_data, threshold=tau
                )
                a["bf_total"] += 1
                if bf.get("correct_branch"):
                    a["bf_correct"] += 1

            if is_terminator:
                a["term_total"] += 1
                if score_termination(response, nodes, threshold=tau)["correct"]:
                    a["term_correct"] += 1

    return acc


def _score_mixed_conversation(
    conv_result: dict[str, Any],
    graph_data: dict[str, Any],
    judge_cache: dict[str, Any],
) -> dict:
    """Score failure modes and contamination for one mixed conversation result.

    Injection-turn classification is delegated to the Haiku judge. Both
    classify_injection_turn() and classify_post_injection_turn() read/write
    judge_cache in-place; caller is responsible for persisting the cache.
    Jaccard is used only for on-procedure recovery and contamination scoring.
    """
    nodes   = graph_data["graph"]["nodes"]
    adj     = _build_adjacency(graph_data)
    acc     = _empty_mixed_acc()
    model   = conv_result.get("model", "unknown")
    conv_id = conv_result.get("conversation_id", "unknown")

    pending_recovery_check = False
    last_injection_category: str | None = None

    for turn in conv_result.get("per_turn", []):
        response      = turn.get("model_response", "")
        utterance     = turn.get("operator_utterance", "")
        gt            = turn.get("ground_truth", {})
        node_type     = gt.get("node_type", "")
        expected_node = gt.get("node_id", "")
        turn_index    = turn.get("turn_index", 0)
        is_inj        = turn.get("is_injection", False)
        post_inj      = turn.get("post_injection", False)
        category      = turn.get("injection_category")

        if is_inj:
            result = classify_injection_turn(
                model_response=response,
                operator_utterance=utterance,
                graph_nodes=nodes,
                model_name=model,
                conv_id=conv_id,
                turn_index=turn_index,
                cache=judge_cache,
            )
            acc["injection_total"] += 1

            if result["fabrication"]:
                acc["fabrication"] += 1
                pending_recovery_check = True
            elif result["forced_mapping"]:
                acc["forced_mapping"] += 1
                pending_recovery_check = True
            else:
                acc["correct_abstention"] += 1
                pending_recovery_check = False
                if expected_node and expected_node in nodes:
                    node_text = nodes[expected_node].get("text", "")
                    if node_text and _text_overlap(response, node_text) >= JACCARD_THRESHOLD_DEFAULT:
                        acc["abstention_with_redirect"] += 1

            last_injection_category = category if category in acc["by_category"] else None

            if category and category in acc["by_category"]:
                cat = acc["by_category"][category]
                cat["total"] += 1
                if result["fabrication"]:
                    cat["fabrication"] += 1
                elif result["forced_mapping"]:
                    cat["forced_mapping"] += 1
                else:
                    cat["correct_abstention"] += 1

        elif node_type != "terminator":
            pt = score_position_tracking(
                response, expected_node, graph_data,
                threshold=JACCARD_THRESHOLD_DEFAULT,
            )

            if pending_recovery_check:
                acc["recovery_total"] += 1
                if pt["correct"]:
                    acc["recovery_correct"] += 1
                pending_recovery_check = False

            if post_inj:
                from src.evaluate.judge import _cache_key as _jck
                _post_key = _jck("post", model, conv_id, turn_index)
                if _post_key in judge_cache:
                    ia = classify_post_injection_turn(
                        model_response=response,
                        operator_utterance=utterance,
                        expected_node_text=nodes.get(expected_node, {}).get("text", "") if expected_node else "",
                        model_name=model,
                        conv_id=conv_id,
                        turn_index=turn_index,
                        cache=judge_cache,
                    )
                    is_ia = ia["inappropriate_abstention"]
                    acc["post_injection_total"] += 1
                    if is_ia:
                        acc["inappropriate_abstention"] += 1
                    if last_injection_category and last_injection_category in acc["by_category"]:
                        acc["by_category"][last_injection_category]["post_total"] += 1
                        if is_ia:
                            acc["by_category"][last_injection_category]["inappropriate_abstention"] += 1
                acc["post_total"] += 1
                if pt["correct"]:
                    acc["post_pt_exact"] += 1
            else:
                acc["pre_total"] += 1
                if pt["correct"]:
                    acc["pre_pt_exact"] += 1

    return acc


def run_analysis(
    evaluation_dir: Path | None = None,
    graph_dir: Path | None = None,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    """Run full analysis on all evaluation result files.

    Returns a dict keyed by model name with nested clean and mixed metrics.
    The judge cache is loaded once, shared across all models, and saved at end.
    """
    if evaluation_dir is None:
        evaluation_dir = EVALUATION_DIR
    if graph_dir is None:
        graph_dir = SYNTHETIC_GRAPHS_DIR
    if thresholds is None:
        thresholds = JACCARD_SWEEP_THRESHOLDS

    graph_cache  = _load_graph_cache(graph_dir)
    judge_cache  = _load_cache()

    result_files = sorted(MODEL_OUTPUTS_DIR.glob("results_*.json"))
    if not result_files:
        return {"error": "No evaluation results found"}

    model_results: dict[str, list[dict]] = {}
    for rf in result_files:
        model_name = " ".join(rf.stem.split("_")[1:])
        try:
            model_results[model_name] = json.loads(rf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"Skipping unreadable file: {rf.name}")

    analysis: dict[str, Any] = {}

    for model_name, results in model_results.items():
        clean_results = [r for r in results if r.get("regime", "clean") == "clean"]
        mixed_results = [r for r in results if r.get("regime") == "mixed"]

        combined_clean = _empty_clean_acc(thresholds)
        for conv in clean_results:
            gd = graph_cache.get(conv.get("graph_id", ""))
            if not gd:
                continue
            per_conv = _score_clean_conversation(conv, gd, thresholds)
            for tau in thresholds:
                for field in combined_clean[tau]:
                    combined_clean[tau][field] += per_conv[tau][field]

        sweep_rates = {tau: _clean_rates(combined_clean[tau]) for tau in thresholds}
        clean_point = sweep_rates[JACCARD_THRESHOLD_DEFAULT]

        combined_mixed = _empty_mixed_acc()
        for conv in mixed_results:
            gd = graph_cache.get(conv.get("graph_id", ""))
            if not gd:
                continue
            per_conv = _score_mixed_conversation(conv, gd, judge_cache)
            for field in ("injection_total", "fabrication", "forced_mapping",
                          "correct_abstention", "abstention_with_redirect",
                          "forced_mapping_nearby", "recovery_total", "recovery_correct",
                          "post_injection_total", "inappropriate_abstention",
                          "pre_pt_exact", "pre_total", "post_pt_exact", "post_total"):
                combined_mixed[field] += per_conv[field]
            for cat in combined_mixed["by_category"]:
                for sub in ("total", "fabrication", "forced_mapping", "correct_abstention",
                            "post_total", "inappropriate_abstention"):
                    combined_mixed["by_category"][cat][sub] += \
                        per_conv["by_category"][cat][sub]

        analysis[model_name] = {
            "model": model_name,
            "total_clean_conversations": len(clean_results),
            "total_mixed_conversations": len(mixed_results),
            "clean": {
                "metrics": {
                    "step_accuracy":           clean_point["position_tracking_exact"],
                    "termination_recognition": clean_point["termination_recognition"],
                    "off_graph_rate":          clean_point["off_graph_rate"],
                },
                "diagnostics": {
                    "position_tracking_forward": clean_point["position_tracking_forward"],
                    "branch_following":          clean_point["branch_following"],
                },
                "threshold_sweep": {str(tau): r for tau, r in sweep_rates.items()},
            },
            "mixed": _mixed_rates(combined_mixed),
        }

    _save_cache(judge_cache)

    report_path = evaluation_dir / "analysis_report.json"
    report_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"Analysis written to {report_path}")

    return analysis


def analyse_injection_density_effect(
    evaluation_dir: Path | None = None,
    conversations_dir: Path | None = None,
    n_quartiles: int = 4,
) -> dict[str, Any]:
    """Compute per-model failure-mode rates bucketed by injection density.

    Injection density = injection_count / num_candidate_positions, read from
    the injection_log stored in each mixed conversation file.

    Conversations are bucketed into n_quartiles equal-width density bins.
    For each model and bucket, fabrication / forced-mapping / correct-abstention
    / inappropriate-abstention rates are computed.

    Writes density_degradation.json to evaluation_dir and returns the dict.
    """
    from src.config import CONVERSATIONS_DIR
    if evaluation_dir is None:
        evaluation_dir = EVALUATION_DIR
    if conversations_dir is None:
        conversations_dir = CONVERSATIONS_DIR

    graph_cache = _load_graph_cache(SYNTHETIC_GRAPHS_DIR)
    judge_cache = _load_cache()

    density_map: dict[str, float] = {}
    for f in sorted(conversations_dir.rglob("*_mixed.json")):
        conv = json.loads(f.read_text(encoding="utf-8"))
        log = conv.get("injection_log", [])
        if log:
            rate = log[0].get("injection_rate")
            num_cands = log[0].get("num_candidates")
            if rate is not None:
                density_map[conv.get("conversation_id", f.stem)] = rate
            elif num_cands and num_cands > 0:
                density_map[conv.get("conversation_id", f.stem)] = len(log) / num_cands

    if not density_map:
        return {"error": "No injection density data found in mixed conversations"}

    rates = sorted(density_map.values())
    n = len(rates)
    boundaries = [rates[int(i * n / n_quartiles)] for i in range(n_quartiles)] + [rates[-1] + 1e-9]

    def _bucket(rate: float) -> int:
        for i, b in enumerate(boundaries[1:]):
            if rate < b:
                return i
        return n_quartiles - 1

    result_files = sorted(MODEL_OUTPUTS_DIR.glob("results_*.json"))
    density_analysis: dict[str, Any] = {}

    for rf in result_files:
        model_name = " ".join(rf.stem.split("_")[1:])
        results = json.loads(rf.read_text(encoding="utf-8"))
        mixed_results = [r for r in results if r.get("regime") == "mixed"]

        buckets: list[dict] = [
            {"injection_total": 0, "fabrication": 0, "forced_mapping": 0,
             "correct_abstention": 0, "post_injection_total": 0, "inappropriate_abstention": 0,
             "density_min": boundaries[i], "density_max": boundaries[i + 1]}
            for i in range(n_quartiles)
        ]

        for conv_result in mixed_results:
            conv_id = conv_result.get("conversation_id", "")
            density = density_map.get(conv_id)
            if density is None:
                continue
            b_idx = _bucket(density)

            graph_id = conv_result.get("graph_id", "")
            gd = graph_cache.get(graph_id)
            if not gd:
                continue
            nodes = gd["graph"]["nodes"]
            model = conv_result.get("model", model_name)

            for turn in conv_result.get("per_turn", []):
                is_inj    = turn.get("is_injection", False)
                post_inj  = turn.get("post_injection", False)
                response  = turn.get("model_response", "")
                utterance = turn.get("operator_utterance", "")
                turn_index = turn.get("turn_index", 0)
                gt         = turn.get("ground_truth", {})
                expected_node = gt.get("node_id", "")

                if is_inj:
                    buckets[b_idx]["injection_total"] += 1
                    result = classify_injection_turn(
                        model_response=response,
                        operator_utterance=utterance,
                        graph_nodes=nodes,
                        model_name=model,
                        conv_id=conv_id,
                        turn_index=turn_index,
                        cache=judge_cache,
                    )
                    if result["fabrication"]:
                        buckets[b_idx]["fabrication"] += 1
                    elif result["forced_mapping"]:
                        buckets[b_idx]["forced_mapping"] += 1
                    else:
                        buckets[b_idx]["correct_abstention"] += 1

                elif post_inj:
                    buckets[b_idx]["post_injection_total"] += 1
                    expected_text = nodes.get(expected_node, {}).get("text", "") if expected_node else ""
                    ia = classify_post_injection_turn(
                        model_response=response,
                        operator_utterance=utterance,
                        expected_node_text=expected_text,
                        model_name=model,
                        conv_id=conv_id,
                        turn_index=turn_index,
                        cache=judge_cache,
                    )
                    if ia["inappropriate_abstention"]:
                        buckets[b_idx]["inappropriate_abstention"] += 1

        def _bucket_rates(bkt: dict) -> dict:
            inj  = bkt["injection_total"]
            post = bkt["post_injection_total"]
            return {
                "density_min": round(bkt["density_min"], 3),
                "density_max": round(bkt["density_max"], 3),
                "fabrication_rate":               bkt["fabrication"]            / inj  if inj  else None,
                "forced_mapping_rate":            bkt["forced_mapping"]         / inj  if inj  else None,
                "correct_abstention_rate":        bkt["correct_abstention"]     / inj  if inj  else None,
                "inappropriate_abstention_rate":  bkt["inappropriate_abstention"] / post if post else None,
                "injection_count":      inj,
                "post_injection_count": post,
            }

        density_analysis[model_name] = {
            "quartile_boundaries": [round(x, 3) for x in boundaries],
            "by_density_quartile": [_bucket_rates(b) for b in buckets],
        }

    _save_cache(judge_cache)

    out_path = evaluation_dir / "density_degradation.json"
    out_path.write_text(json.dumps(density_analysis, indent=2), encoding="utf-8")
    print(f"Density degradation analysis written to {out_path}")
    return density_analysis


def compute_injection_dynamics(
    evaluation_dir: Path | None = None,
    graph_dir: Path | None = None,
) -> dict[str, Any]:
    """Collect per-injection-turn outcomes and per-conversation SA for dynamics figures.

    For each model and mixed conversation records:
      - num_proc_turns: procedure length from matching clean conversation
      - sa_rate: branch-following accuracy on matching clean conversation
      - injections: list of {index, outcome} in conversation order

    Writes injection_dynamics.json to evaluation_dir and returns the dict.
    """
    if evaluation_dir is None:
        evaluation_dir = EVALUATION_DIR
    if graph_dir is None:
        graph_dir = SYNTHETIC_GRAPHS_DIR

    graph_cache = _load_graph_cache(graph_dir)
    judge_cache = _load_cache()

    result_files = sorted(MODEL_OUTPUTS_DIR.glob("results_*.json"))
    if not result_files:
        return {"error": "No evaluation results found"}

    dynamics: dict[str, Any] = {}
    out_path = evaluation_dir / "injection_dynamics.json"

    for rf in result_files:
        model_name = " ".join(rf.stem.split("_")[1:])
        results = json.loads(rf.read_text(encoding="utf-8"))

        # Clean pass: SA rate and num_turns per conversation
        clean_meta: dict[str, dict] = {}
        for conv in results:
            if conv.get("regime") != "clean":
                continue
            gd = graph_cache.get(conv.get("graph_id", ""))
            if not gd:
                continue
            per_conv = _score_clean_conversation(conv, gd, [JACCARD_THRESHOLD_DEFAULT])
            acc = per_conv[JACCARD_THRESHOLD_DEFAULT]
            sa = acc["bf_correct"] / acc["bf_total"] if acc["bf_total"] else None
            clean_meta[conv["conversation_id"]] = {
                "num_turns": conv.get("num_turns", 0),
                "sa_rate": sa,
            }

        # Mixed pass: per-injection outcomes
        mixed_convs = [r for r in results if r.get("regime") == "mixed"]
        total_mixed = len(mixed_convs)
        print(f"\n{model_name}  ({total_mixed} mixed conversations)")
        conversations = []
        for conv_i, conv in enumerate(mixed_convs, 1):
            conv_id  = conv.get("conversation_id", "")
            graph_id = conv.get("graph_id", "")
            gd = graph_cache.get(graph_id)
            if not gd:
                continue
            nodes = gd["graph"]["nodes"]
            model = conv.get("model", "")

            clean_id = conv_id.replace("_mixed", "")
            meta = clean_meta.get(clean_id, {})

            injections = []
            for turn in conv.get("per_turn", []):
                if not turn.get("is_injection", False):
                    continue
                result = classify_injection_turn(
                    model_response=turn.get("model_response", ""),
                    operator_utterance=turn.get("operator_utterance", ""),
                    graph_nodes=nodes,
                    model_name=model,
                    conv_id=conv_id,
                    turn_index=turn.get("turn_index", 0),
                    cache=judge_cache,
                )
                injections.append({
                    "index": len(injections),
                    "outcome": result["outcome"],
                })

            conv_meta = conv.get("metadata", {})
            if injections:
                conversations.append({
                    "conv_id": conv_id,
                    "num_proc_turns": meta.get("num_turns", 0),
                    "sa_rate": meta.get("sa_rate"),
                    "num_decisions": conv_meta.get("num_decisions"),
                    "graph_depth": conv_meta.get("depth"),
                    "injections": injections,
                })

            print(f"  [{conv_i}/{total_mixed}] {conv_id}  "
                  f"({len(injections)} inj, cache={sum(1 for i in injections if i)})",
                  end="\r", flush=True)

        print()  # newline after progress line
        dynamics[model_name] = {"conversations": conversations}
        print(f"  done: {len(conversations)} conversations scored")

        # Save after every model so the run is resumable if interrupted
        _save_cache(judge_cache)
        out_path.write_text(json.dumps(dynamics, indent=2), encoding="utf-8")

    print(f"Injection dynamics written to {out_path}")
    return dynamics


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "dynamics":
        compute_injection_dynamics()
        sys.exit(0)
    res = run_analysis()
    if "error" in res:
        print(res["error"])
        sys.exit(1)
    for model, val in res.items():
        c = val["clean"]["metrics"]
        d = val["clean"]["diagnostics"]
        m = val["mixed"]

        def _pct(v): return f"{v:.1%}" if v is not None else "n/a"

        print(f"\n{model}")
        print(f"  On-procedure  — SA: {_pct(c['step_accuracy'])} | Term: {_pct(c['termination_recognition'])}")
        print(f"  (Diagnostic) Branch following: {_pct(d['branch_following'])}")
        print(f"  Injection  — Fabrication: {_pct(m['fabrication_rate'])} | "
              f"Forced: {_pct(m['forced_mapping_rate'])} | "
              f"Abstention: {_pct(m['correct_abstention_rate'])}")
        print(f"  Abstention redirect: {_pct(m['abstention_redirect_rate'])} | "
              f"Forced nearby: {_pct(m['forced_mapping_nearby_rate'])} | "
              f"Recovery: {_pct(m['recovery_rate'])}")
        delta = m['contamination']['delta']
        print(f"  Contamination Δ: {delta:+.3f}" if delta is not None else "  Contamination: n/a")
