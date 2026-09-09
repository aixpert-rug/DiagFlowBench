"""Batch API injection pipeline using the Anthropic Message Batches API.

Three-phase workflow: Phase 1 plans injection positions and submits generation
requests; Phase 2 collects utterances and submits a verification batch; Phase 3
collects verification results, retries failures synchronously, and writes mixed
conversation JSONs. State is persisted to batch_injection_state.json.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import anthropic

from src.config import (
    ANTHROPIC_API_KEY,
    CONVERSATIONS_DIR,
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
    INJECTION_RATE_MIN,
    INJECTION_RATE_MAX,
    SYNTHETIC_GRAPHS_DIR,
    VERIFICATION_MODEL,
    VERIFICATION_TEMPERATURE,
)
from src.conversations.batch_generator import (
    _extract_anthropic_model_id,
    _parse_json_response,
    _poll_until_done,
)
from src.conversations.injection_generator import (
    INJECTION_CATEGORIES,
    _INJECTION_SYSTEM,
    _INJECTION_VERIFIER_SYSTEM,
    _candidate_positions,
    _outgoing_edge_labels,
    generate_injection_utterance,
    verify_injection_utterance,
)

_BATCH_INJECTION_STATE_PATH = CONVERSATIONS_DIR.parent / "batch_injection_state.json"

_ANTHROPIC_GENERATION_MODEL = _extract_anthropic_model_id(GENERATION_MODEL)
_ANTHROPIC_VERIFICATION_MODEL = _extract_anthropic_model_id(VERIFICATION_MODEL)



def _get_client() -> anthropic.Anthropic:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY must be set in .env for the Batch API.")
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _save_state(state: dict) -> None:
    _BATCH_INJECTION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _BATCH_INJECTION_STATE_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _load_state() -> dict:
    if _BATCH_INJECTION_STATE_PATH.exists():
        return json.loads(_BATCH_INJECTION_STATE_PATH.read_text(encoding="utf-8"))
    return {}


def _load_clean_conversations(conversations_dir: Path) -> list[tuple[str, dict]]:
    """Load all clean (non-mixed) conversations, sorted for determinism."""
    result = []
    for f in sorted(conversations_dir.glob("*.json")):
        if "_mixed" in f.stem or f.stem == "quality_report":
            continue
        try:
            conv = json.loads(f.read_text(encoding="utf-8"))
            result.append((f.stem, conv))
        except json.JSONDecodeError:
            print(f"  WARNING: Could not read {f.name} — skipping.")
    return result


def _load_graph_cache(graph_dir: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    for p in sorted(graph_dir.glob("GRAPH*.json")):
        g = json.loads(p.read_text(encoding="utf-8"))
        cache[g["id"]] = g
    return cache



def submit_injection_generation_batch(
    conversations_dir: Path | None = None,
    graph_dir: Path | None = None,
    seed: int = 42,
) -> str:
    """Plan injections for all clean conversations and submit a generation batch.

    Injection positions and categories are sampled deterministically using the
    provided seed and a sorted file iteration order. The plan is saved to
    batch_injection_state.json so Phase 2 can reconstruct it without the
    original conversation files.

    Returns the batch ID.
    """
    if conversations_dir is None:
        conversations_dir = CONVERSATIONS_DIR
    if graph_dir is None:
        graph_dir = SYNTHETIC_GRAPHS_DIR

    conversations = _load_clean_conversations(conversations_dir)
    graph_cache = _load_graph_cache(graph_dir)
    client = _get_client()
    rng = random.Random(seed)

    injection_plan: dict[str, Any] = {}
    requests: list[dict] = []

    for conv_stem, conv in conversations:
        graph_id = conv.get("graph_id", "")
        graph_data = graph_cache.get(graph_id)
        if not graph_data:
            print(f"  WARNING: graph {graph_id} not found for {conv_stem} — skipping.")
            continue

        turns = conv.get("turns", [])
        nodes = graph_data["graph"]["nodes"]
        candidates = _candidate_positions(turns)

        if not candidates:
            injection_plan[conv_stem] = {
                "graph_id": graph_id,
                "conv_id": conv.get("conversation_id", conv_stem),
                "k": 0,
                "positions": {},
            }
            continue

        # Proportional injection count: rate sampled per conversation
        rate = rng.uniform(INJECTION_RATE_MIN, INJECTION_RATE_MAX)
        k = max(1, round(len(candidates) * rate))
        k = min(k, len(candidates))
        positions = sorted(rng.sample(candidates, k))

        plan_positions: dict[str, Any] = {}
        for pos_idx, pos in enumerate(positions):
            turn = turns[pos]
            node_id = turn.get("node_id", "")
            node_text = nodes.get(node_id, {}).get("text", "")
            edge_labels = _outgoing_edge_labels(node_id, graph_data)
            # Round-robin categories for balanced distribution
            category = INJECTION_CATEGORIES[pos_idx % len(INJECTION_CATEGORIES)]

            plan_positions[str(pos)] = {
                "node_id": node_id,
                "node_type": turn.get("node_type", ""),
                "node_text": node_text,
                "edge_labels": edge_labels,
                "category": category,
                "injection_rate": round(rate, 4),
                "num_candidates": len(candidates),
            }

            # Generation request: one per (conversation, position) pair.
            # custom_id format: gen__{conv_stem}__{pos}
            # conv_stem may contain underscores, so rsplit("__", 1) is safe.
            user_content = json.dumps({
                "current_node_text": node_text,
                "outgoing_edge_labels": edge_labels,
                "failure_category": category,
            }, indent=2)

            requests.append({
                "custom_id": f"gen__{conv_stem}__{pos}",
                "params": {
                    "model": _ANTHROPIC_GENERATION_MODEL,
                    "max_tokens": 512,
                    "temperature": GENERATION_TEMPERATURE,
                    "system": _INJECTION_SYSTEM,
                    "messages": [{"role": "user", "content": user_content}],
                },
            })

        injection_plan[conv_stem] = {
            "graph_id": graph_id,
            "conv_id": conv.get("conversation_id", conv_stem),
            "k": k,
            "positions": plan_positions,
        }

    if not requests:
        raise RuntimeError(
            "No generation requests to submit. "
            "Are there clean conversations in the conversations directory?"
        )

    print(
        f"Submitting injection generation batch: {len(requests)} requests "
        f"across {sum(1 for p in injection_plan.values() if p['k'] > 0)} conversations..."
    )
    batch = client.beta.messages.batches.create(requests=requests)
    print(f"Batch submitted. ID: {batch.id}")
    print(f"Status: {batch.processing_status}")

    state = _load_state()
    state["gen_batch_id"] = batch.id
    state["injection_plan"] = injection_plan
    state["conversations_dir"] = str(conversations_dir)
    state["graph_dir"] = str(graph_dir)
    state["seed"] = seed
    state["gen_submitted_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _save_state(state)

    return batch.id



def collect_gen_and_submit_verification(batch_id: str | None = None) -> str:
    """Poll the generation batch, collect utterances, submit a verification batch.

    Returns the verification batch ID.
    """
    state = _load_state()
    if batch_id is None:
        batch_id = state.get("gen_batch_id")
    if not batch_id:
        raise ValueError(
            "No generation batch ID found in state. Run submit-gen first."
        )

    injection_plan: dict[str, Any] = state.get("injection_plan", {})
    client = _get_client()

    _poll_until_done(client, batch_id, label="injection-generation")

    gen_results: dict[str, dict[str, str]] = {}
    failed = 0

    for result in client.beta.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            print(f"  WARNING: {result.custom_id} failed — position will be skipped.")
            failed += 1
            continue

        content_blocks = result.result.message.content
        if not content_blocks:
            print(f"  WARNING: Empty content for {result.custom_id} — skipping.")
            failed += 1
            continue

        # Parse custom_id: gen__{conv_stem}__{pos}
        # rsplit("__", 1) splits once from the right, giving
        # [conv_stem, pos] even if conv_stem has single underscores.
        try:
            without_prefix = result.custom_id.removeprefix("gen__")
            parts = without_prefix.rsplit("__", 1)
            if len(parts) != 2:
                raise ValueError(f"Expected 2 parts, got {len(parts)}")
            conv_stem, pos_str = parts
        except Exception as exc:
            print(f"  WARNING: Could not parse custom_id {result.custom_id!r}: {exc}")
            continue

        try:
            parsed = _parse_json_response(content_blocks[0].text, result.custom_id)
            utterance = parsed.get("utterance", "") if isinstance(parsed, dict) else ""
            if utterance:
                gen_results.setdefault(conv_stem, {})[pos_str] = utterance
            else:
                print(f"  WARNING: Empty utterance for {result.custom_id} — skipping.")
                failed += 1
        except Exception as exc:
            print(f"  WARNING: Parse error for {result.custom_id}: {exc}")
            failed += 1

    total_gen = sum(p["k"] for p in injection_plan.values())
    collected = sum(len(v) for v in gen_results.values())
    print(f"Generation complete: {collected}/{total_gen} utterances collected "
          f"({failed} failed).")

    # custom_id format: ver__{conv_stem}__{pos_str}
    verify_requests: list[dict] = []
    for conv_stem, positions_gen in gen_results.items():
        conv_plan = injection_plan.get(conv_stem, {})
        for pos_str, utterance in positions_gen.items():
            pos_plan = conv_plan.get("positions", {}).get(pos_str)
            if not pos_plan:
                continue

            user_content = json.dumps({
                "current_node_text": pos_plan["node_text"],
                "outgoing_edge_labels": pos_plan["edge_labels"],
                "utterance": utterance,
            }, indent=2)

            verify_requests.append({
                "custom_id": f"ver__{conv_stem}__{pos_str}",
                "params": {
                    "model": _ANTHROPIC_VERIFICATION_MODEL,
                    "max_tokens": 256,
                    "temperature": VERIFICATION_TEMPERATURE,
                    "system": _INJECTION_VERIFIER_SYSTEM,
                    "messages": [{"role": "user", "content": user_content}],
                },
            })

    if not verify_requests:
        raise RuntimeError(
            "No verification requests to submit — all generation calls failed."
        )

    print(f"Submitting verification batch: {len(verify_requests)} requests...")
    batch2 = client.beta.messages.batches.create(requests=verify_requests)
    print(f"Verification batch submitted. ID: {batch2.id}")

    state["ver_batch_id"] = batch2.id
    state["gen_results"] = gen_results
    state["ver_submitted_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _save_state(state)

    return batch2.id



def collect_verify_and_assemble(
    batch_id: str | None = None,
    output_dir: Path | None = None,
    skip_existing: bool = True,
    max_sync_retries: int = 2,
) -> list[Path]:
    """Poll the verification batch, handle failures, write mixed conversations.

    Utterances that pass verification are used as-is. Utterances that fail
    verification receive up to max_sync_retries synchronous retry attempts
    (using the OpenAI-compatible Anthropic client). Positions where all retries
    fail are dropped silently; a conversation with zero successful injections
    still produces a mixed-annotated file (regime="mixed", injection_count=0).

    Returns list of written file paths.
    """
    state = _load_state()
    if batch_id is None:
        batch_id = state.get("ver_batch_id")
    if not batch_id:
        raise ValueError(
            "No verification batch ID found in state. Run collect-gen first."
        )

    injection_plan: dict[str, Any] = state.get("injection_plan", {})
    gen_results: dict[str, dict[str, str]] = state.get("gen_results", {})
    conversations_dir = Path(state.get("conversations_dir", str(CONVERSATIONS_DIR)))
    graph_dir = Path(state.get("graph_dir", str(SYNTHETIC_GRAPHS_DIR)))

    if output_dir is None:
        output_dir = conversations_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    client = _get_client()
    graph_cache = _load_graph_cache(graph_dir)

    _poll_until_done(client, batch_id, label="injection-verification")

    ver_results: dict[str, dict[str, bool]] = {}

    for result in client.beta.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            print(f"  WARNING: {result.custom_id} verification call failed — "
                  f"will attempt sync retry.")
            continue

        content_blocks = result.result.message.content
        if not content_blocks:
            continue

        # Parse custom_id: ver__{conv_stem}__{pos_str}
        # rsplit("__", 1) splits once from the right: [conv_stem, pos_str]
        try:
            without_prefix = result.custom_id.removeprefix("ver__")
            conv_stem, pos_str = without_prefix.rsplit("__", 1)
        except ValueError:
            print(f"  WARNING: Could not parse verify custom_id {result.custom_id!r}")
            continue

        try:
            parsed = _parse_json_response(content_blocks[0].text, result.custom_id)
            valid = bool(parsed.get("valid", False)) if isinstance(parsed, dict) else False
            ver_results.setdefault(conv_stem, {})[pos_str] = valid
        except Exception as exc:
            print(f"  WARNING: Parse error for {result.custom_id}: {exc}")

    conversations = _load_clean_conversations(conversations_dir)
    conv_map = {stem: conv for stem, conv in conversations}

    # Lazy-init OpenAI-compatible client for sync retries
    openai_client: Any = None

    output_files: list[Path] = []
    total = len(injection_plan)

    for idx, (conv_stem, plan) in enumerate(injection_plan.items()):
        out_path = output_dir / f"{conv_stem}_mixed.json"

        if skip_existing and out_path.exists():
            output_files.append(out_path)
            continue

        conv = conv_map.get(conv_stem)
        if not conv:
            print(f"  WARNING: Conversation file not found for {conv_stem} — skipping.")
            continue

        graph_id = plan.get("graph_id", "")
        graph_data = graph_cache.get(graph_id)
        if not graph_data:
            print(f"  WARNING: Graph {graph_id} not found for {conv_stem} — skipping.")
            continue

        nodes = graph_data["graph"]["nodes"]
        base_turns = conv.get("turns", [])
        conv_id = conv.get("conversation_id", conv_stem)
        plan_positions = plan.get("positions", {})

        valid_injections: dict[int, dict] = {}

        for pos_str, pos_plan in plan_positions.items():
            pos = int(pos_str)
            utterance = gen_results.get(conv_stem, {}).get(pos_str)
            if not utterance:
                continue  # Generation failed outright — no utterance to verify

            verified = ver_results.get(conv_stem, {}).get(pos_str)

            if verified:
                valid_injections[pos] = {
                    "utterance": utterance,
                    "category": pos_plan["category"],
                    "node_id": pos_plan["node_id"],
                    "node_type": pos_plan["node_type"],
                    "node_text": pos_plan["node_text"],
                    "edge_labels": pos_plan["edge_labels"],
                }

            else:
                # Verification failed (False or missing) — sync retry.
                if openai_client is None:
                    from src.conversations.graph_generator import _get_client as _get_openai_client
                    openai_client = _get_openai_client()

                retry_utterance: str | None = None
                for attempt in range(1, max_sync_retries + 1):
                    try:
                        candidate = generate_injection_utterance(
                            openai_client,
                            node_text=pos_plan["node_text"],
                            edge_labels=pos_plan["edge_labels"],
                            category=pos_plan["category"],
                        )
                        ok, _ = verify_injection_utterance(
                            openai_client,
                            node_text=pos_plan["node_text"],
                            edge_labels=pos_plan["edge_labels"],
                            utterance=candidate,
                        )
                        if ok:
                            retry_utterance = candidate
                            break
                    except Exception as exc:
                        print(f"  WARNING: Sync retry {attempt}/{max_sync_retries} "
                              f"for {conv_stem}[{pos}]: {exc}")

                if retry_utterance:
                    valid_injections[pos] = {
                        "utterance": retry_utterance,
                        "category": pos_plan["category"],
                        "node_id": pos_plan["node_id"],
                        "node_type": pos_plan["node_type"],
                        "node_text": pos_plan["node_text"],
                        "edge_labels": pos_plan["edge_labels"],
                    }
                else:
                    print(f"  WARNING: All retries exhausted for {conv_stem}[{pos}] "
                          f"— position dropped.")

        # Splice valid injections into the turn list.
        sorted_positions = sorted(valid_injections.keys())
        first_injection_pos: int | None = sorted_positions[0] if sorted_positions else None
        injection_log: list[dict] = []
        injection_map: dict[int, dict] = {}

        for pos in sorted_positions:
            inj = valid_injections[pos]
            injection_map[pos] = {
                "node_id": inj["node_id"],
                "node_type": inj["node_type"],
                "speaker": "operator",
                "utterance": inj["utterance"],
                "edge_label_to_next": None,
                "is_injection": True,
                "injection_category": inj["category"],
            }
            injection_log.append({
                "position": pos,
                "node_id": inj["node_id"],
                "node_text": inj["node_text"],
                "edge_labels": inj["edge_labels"],
                "category": inj["category"],
                "utterance": inj["utterance"],
                "injection_rate": inj.get("injection_rate"),
                "num_candidates": inj.get("num_candidates"),
            })

        mixed_turns: list[dict] = []
        for i, turn in enumerate(base_turns):
            post = first_injection_pos is not None and i >= first_injection_pos

            if i in injection_map:
                inj_turn = dict(injection_map[i])
                inj_turn["post_injection"] = post
                mixed_turns.append(inj_turn)

            annotated = dict(turn)
            annotated["is_injection"] = False
            annotated["injection_category"] = None
            annotated["post_injection"] = post
            mixed_turns.append(annotated)

        mixed_conv = {
            **conv,
            "conversation_id": f"{conv_id}_mixed",
            "regime": "mixed",
            "turns": mixed_turns,
            "injection_log": injection_log,
        }
        mixed_conv.setdefault("metadata", {})["injection_count"] = len(injection_log)

        out_path.write_text(
            json.dumps(mixed_conv, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        output_files.append(out_path)

        if (idx + 1) % 100 == 0:
            print(f"  [{idx + 1}/{total}] assembled...")

    print(f"\nAssembly complete: {len(output_files)} mixed conversations "
          f"written to {output_dir}")
    return output_files
