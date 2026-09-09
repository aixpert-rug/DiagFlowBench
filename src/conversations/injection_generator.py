"""Mixed conversation generation for DiagFlowBench.

For each base conversation, injects k off-procedure operator turns at randomly
sampled non-terminator positions. k is proportional to candidate positions
(k = max(1, round(candidates * rate)), rate ~ Uniform[0.05, 0.30]). Categories
cycle round-robin for balanced distribution; each injection is verified to not
entail any outgoing edge of the current node.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import yaml

from src.conversations.graph_generator import (
    _call_llm,
    _get_client,
    _API_GENERATION_MODEL,
    _API_VERIFICATION_MODEL,
)
from src.config import (
    GENERATION_TEMPERATURE,
    VERIFICATION_TEMPERATURE,
    INJECTION_RATE_MIN,
    INJECTION_RATE_MAX,
)

_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "prompts" / "generation.yaml"

INJECTION_CATEGORIES: list[str] = [
    "coverage_gap",
    "undocumented_malfunction",
    "unrelated_question",
]


def _load_injection_prompts() -> tuple[str, str]:
    prompts = yaml.safe_load(_PROMPTS_PATH.read_text(encoding="utf-8"))
    missing = {"injection_system", "injection_verifier_system"} - set(prompts)
    if missing:
        raise ValueError(f"generation.yaml missing keys: {missing}")
    return prompts["injection_system"], prompts["injection_verifier_system"]


_INJECTION_SYSTEM, _INJECTION_VERIFIER_SYSTEM = _load_injection_prompts()



def _outgoing_edge_labels(node_id: str, graph_data: dict) -> list[str]:
    """Return the outgoing edge labels for a node (empty list if none)."""
    return [
        e["label"]
        for e in graph_data["graph"]["edges"]
        if e["source"] == node_id and e.get("label")
    ]


def _candidate_positions(turns: list[dict]) -> list[int]:
    """Indices of operator turns that are not at terminator nodes.

    Only operator turns are candidates because injections are operator
    utterances; terminator turns are excluded because they mark procedure
    endpoints where injection would be structurally incoherent.
    """
    return [
        i for i, t in enumerate(turns)
        if t.get("speaker") == "operator"
        and t.get("node_type") != "terminator"
    ]



def generate_injection_utterance(
    client: Any,
    *,
    node_text: str,
    edge_labels: list[str],
    category: str,
) -> str:
    """Generate one off-procedure operator utterance for a given category."""
    user_prompt = json.dumps({
        "current_node_text": node_text,
        "outgoing_edge_labels": edge_labels,
        "failure_category": category,
    }, indent=2)

    result = _call_llm(
        client,
        model=_API_GENERATION_MODEL,
        system_prompt=_INJECTION_SYSTEM,
        user_prompt=user_prompt,
        temperature=GENERATION_TEMPERATURE,
        phase="injection_generation",
    )

    if isinstance(result, dict):
        utterance = result.get("utterance", "")
        if utterance:
            return utterance
    raise ValueError(f"injection_generation returned no utterance: {result!r}")


def verify_injection_utterance(
    client: Any,
    *,
    node_text: str,
    edge_labels: list[str],
    utterance: str,
) -> tuple[bool, str]:
    """Verify that an utterance does not entail any outgoing edge label.

    Returns (valid, reason). valid=True means the utterance is genuinely
    off-procedure and safe to use as an injection.
    """
    user_prompt = json.dumps({
        "current_node_text": node_text,
        "outgoing_edge_labels": edge_labels,
        "utterance": utterance,
    }, indent=2)

    result = _call_llm(
        client,
        model=_API_VERIFICATION_MODEL,
        system_prompt=_INJECTION_VERIFIER_SYSTEM,
        user_prompt=user_prompt,
        temperature=VERIFICATION_TEMPERATURE,
        phase="injection_verification",
    )

    if isinstance(result, dict):
        return bool(result.get("valid", False)), result.get("reason", "")
    return False, f"Unexpected verifier format: {result!r}"



def generate_mixed_conversation(
    client: Any,
    base_turns: list[dict],
    graph_data: dict,
    *,
    rng: random.Random | None = None,
    delay: float = 0.5,
    max_attempts_per_injection: int = 3,
) -> tuple[list[dict], list[dict]]:
    """Inject k off-procedure turns into a clean base conversation.

    Each injection is inserted BEFORE the operator turn at the sampled position.
    Returns (mixed_turns, injection_log).
    """
    if rng is None:
        rng = random.Random()

    nodes = graph_data["graph"]["nodes"]
    candidates = _candidate_positions(base_turns)

    if not candidates:
        # No eligible positions — return clean turns with annotation only
        annotated = [
            {**t, "is_injection": False, "injection_category": None, "post_injection": False}
            for t in base_turns
        ]
        return annotated, []

    # Proportional injection count: rate sampled per conversation, bounded to [1, len(candidates)]
    rate = rng.uniform(INJECTION_RATE_MIN, INJECTION_RATE_MAX)
    k = max(1, round(len(candidates) * rate))
    k = min(k, len(candidates))
    positions = sorted(rng.sample(candidates, k))
    first_injection_pos = positions[0]

    injection_log: list[dict] = []
    injection_map: dict[int, dict] = {}   # position → injection turn dict

    for pos_idx, pos in enumerate(positions):
        turn = base_turns[pos]
        node_id = turn.get("node_id", "")
        node_text = nodes.get(node_id, {}).get("text", "")
        edge_labels = _outgoing_edge_labels(node_id, graph_data)
        # Round-robin categories for balanced distribution across positions
        category = INJECTION_CATEGORIES[pos_idx % len(INJECTION_CATEGORIES)]

        utterance: str | None = None
        attempts = 0

        for attempt in range(1, max_attempts_per_injection + 1):
            attempts = attempt
            try:
                candidate = generate_injection_utterance(
                    client,
                    node_text=node_text,
                    edge_labels=edge_labels,
                    category=category,
                )
                time.sleep(delay)

                valid, reason = verify_injection_utterance(
                    client,
                    node_text=node_text,
                    edge_labels=edge_labels,
                    utterance=candidate,
                )
                time.sleep(delay)

                if valid:
                    utterance = candidate
                    break
                # Rejected — retry with same category; only the specific utterance was too close to an edge label

            except Exception as e:
                if attempt == max_attempts_per_injection:
                    print(f"  WARNING: injection at pos {pos} failed after "
                          f"{attempt} attempts: {e}")

        if utterance is None:
            # Exhausted retries — skip this position silently
            continue

        injection_map[pos] = {
            "node_id": node_id,
            "node_type": turn.get("node_type", ""),
            "speaker": "operator",
            "utterance": utterance,
            "edge_label_to_next": None,
            "is_injection": True,
            "injection_category": category,
        }
        injection_log.append({
            "position": pos,
            "node_id": node_id,
            "node_text": node_text,
            "edge_labels": edge_labels,
            "category": category,
            "utterance": utterance,
            "generation_attempts": attempts,
            "injection_rate": round(rate, 4),
            "num_candidates": len(candidates),
        })

    # Splice injections into the turn list
    # post_injection=True from the injection position onwards (inclusive),
    # since the on-procedure turn at that index is the first potentially contaminated turn.
    mixed_turns: list[dict] = []
    for i, turn in enumerate(base_turns):
        if i in injection_map:
            inj = dict(injection_map[i])
            inj["post_injection"] = i >= first_injection_pos
            mixed_turns.append(inj)

        annotated = dict(turn)
        annotated["is_injection"] = False
        annotated["injection_category"] = None
        annotated["post_injection"] = i >= first_injection_pos
        mixed_turns.append(annotated)

    return mixed_turns, injection_log



def run_injection_pipeline(
    *,
    conversations_dir: Path,
    output_dir: Path,
    graph_dir: Path,
    seed: int = 42,
    skip_existing: bool = True,
    delay: float = 0.5,
) -> list[Path]:
    """Generate mixed versions of all base conversations in conversations_dir.

    Only processes files whose names do NOT contain '_mixed' (avoids
    re-injecting already-mixed conversations). Writes output to output_dir
    as {conversation_id}_mixed.json.

    Returns list of written paths.
    """
    import json as _json
    from src.config import CONVERSATIONS_DIR, SYNTHETIC_GRAPHS_DIR

    if output_dir is None:
        output_dir = conversations_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_cache: dict[str, dict] = {}
    for p in sorted(graph_dir.glob("GRAPH*.json")):
        g = _json.loads(p.read_text(encoding="utf-8"))
        graph_cache[g["id"]] = g

    client = _get_client()
    rng = random.Random(seed)

    conv_files = [
        f for f in sorted(conversations_dir.glob("*.json"))
        if "_mixed" not in f.stem and f.stem != "quality_report"
    ]

    output_files: list[Path] = []
    total = len(conv_files)

    for idx, conv_file in enumerate(conv_files):
        conv = _json.loads(conv_file.read_text(encoding="utf-8"))
        conv_id = conv.get("conversation_id", conv_file.stem)
        graph_id = conv.get("graph_id", "")
        graph_data = graph_cache.get(graph_id)

        out_path = output_dir / f"{conv_file.stem}_mixed.json"

        if skip_existing and out_path.exists():
            output_files.append(out_path)
            continue

        if not graph_data:
            print(f"  WARNING: graph {graph_id} not found for {conv_id}, skipping")
            continue

        print(f"[{idx + 1}/{total}] Injecting {conv_id}...")

        try:
            mixed_turns, injection_log = generate_mixed_conversation(
                client,
                conv["turns"],
                graph_data,
                rng=rng,
                delay=delay,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        mixed_conv = {
            **conv,
            "conversation_id": f"{conv_id}_mixed",
            "regime": "mixed",
            "turns": mixed_turns,
            "injection_log": injection_log,
        }
        mixed_conv.setdefault("metadata", {})["injection_count"] = len(injection_log)

        out_path.write_text(
            _json.dumps(mixed_conv, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        output_files.append(out_path)

    print(f"\nInjection complete: {len(output_files)} mixed conversations written")
    return output_files
