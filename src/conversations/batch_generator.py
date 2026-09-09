"""Batch API conversation generation using the Anthropic Message Batches API.

Three-phase workflow: Phase 1 submits Pass-1 generation requests; Phase 2 polls
Pass-1 results and submits Pass-2 (verification + robustness variants); Phase 3
polls Pass-2 results and assembles final conversation JSONs. Each phase saves
its batch ID to batch_state.json so --batch-id can be omitted on the same machine.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import anthropic

from src.config import (
    ANTHROPIC_API_KEY,
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
    VERIFICATION_MODEL,
    VERIFICATION_TEMPERATURE,
    SYNTHETIC_GRAPHS_DIR,
    PATHS_DIR,
    CONVERSATIONS_DIR,
)
from src.conversations.graph_generator import (
    SYSTEM_PROMPT_PASS1,
    SYSTEM_PROMPT_PASS2,
    ROBUSTNESS_SYSTEM_PROMPT,
    TRANSFORMATIONS,
    assemble_conversation,
    _validate_conversation_structure,
)

_BATCH_STATE_PATH = PATHS_DIR.parent / "batch_state.json"

# Poll settings for batch status checks.
# Anthropic guarantees completion within 24 hours; we enforce this as a hard timeout.
_POLL_TIMEOUT_SECONDS: int = 24 * 3600
_POLL_INITIAL_INTERVAL: int = 60    # seconds
_POLL_BACKOFF_FACTOR: int = 2
_POLL_MAX_INTERVAL: int = 300       # seconds


def _extract_anthropic_model_id(model_str: str) -> str:
    """Strip provider prefix from a model string for the native Anthropic client.

    'anthropic/claude-sonnet-4-20250514' → 'claude-sonnet-4-20250514'
    'claude-sonnet-4-20250514'           → 'claude-sonnet-4-20250514'

    Warns if the prefix is present but not 'anthropic/', since that likely
    means a non-Anthropic model was configured for generation.
    """
    if "/" not in model_str:
        return model_str
    provider, model_id = model_str.split("/", 1)
    if provider != "anthropic":
        import warnings
        warnings.warn(
            f"GENERATION_MODEL has provider prefix '{provider}/' — expected 'anthropic/'. "
            f"Using '{model_id}' as the model ID for the Anthropic Batch API.",
            stacklevel=2,
        )
    return model_id


_ANTHROPIC_GENERATION_MODEL = _extract_anthropic_model_id(GENERATION_MODEL)
_ANTHROPIC_VERIFICATION_MODEL = _extract_anthropic_model_id(VERIFICATION_MODEL)


def _get_client() -> anthropic.Anthropic:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY must be set in .env for Batch API")
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _save_state(state: dict) -> None:
    _BATCH_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _BATCH_STATE_PATH.write_text(json.dumps(state, indent=2))


def _load_state() -> dict:
    if _BATCH_STATE_PATH.exists():
        return json.loads(_BATCH_STATE_PATH.read_text())
    return {}


def _load_graphs() -> dict[str, dict]:
    cache: dict[str, dict] = {}
    for p in sorted(SYNTHETIC_GRAPHS_DIR.glob("GRAPH*.json")):
        g = json.loads(p.read_text(encoding="utf-8"))
        cache[g["id"]] = g
    return cache


def _validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate that the manifest has the expected structure."""
    if "paths" not in manifest:
        raise ValueError("Manifest is missing the 'paths' key.")
    required = {
        "path_id", "graph_id", "node_sequence", "edge_labels", "decision_points",
        "depth", "num_decisions", "num_processes", "num_documents",
        "terminator_type", "terminator_text", "true_terminator",
    }
    for i, path in enumerate(manifest["paths"]):
        missing = required - set(path.keys())
        if missing:
            raise ValueError(
                f"Path record at index {i} (path_id={path.get('path_id', '?')!r}) "
                f"is missing required fields: {missing}"
            )


def _validate_graph_cache(
    graph_cache: dict[str, dict],
    paths: list[dict[str, Any]],
) -> None:
    """Raise ValueError if any path references a graph not in the cache."""
    missing = sorted({p["graph_id"] for p in paths} - set(graph_cache))
    if missing:
        raise ValueError(
            f"The following graph IDs are referenced in the manifest but not "
            f"found in {SYNTHETIC_GRAPHS_DIR}:\n  " + "\n  ".join(missing)
        )


def _parse_json_response(text: str, custom_id: str) -> Any:
    """Parse JSON from an LLM response, with fallback for code-fenced output.

    The Anthropic Batch API does not support response_format enforcement,
    so models may wrap JSON in markdown code fences.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    raise ValueError(
        f"Could not parse JSON for {custom_id!r}. "
        f"Response preview: {text[:300]!r}"
    )


def _parse_turns(parsed: Any, custom_id: str) -> list[dict[str, Any]]:
    """Extract a turn list from a parsed response dict or list.

    Raises ValueError on unexpected structure so the caller can skip or log.
    """
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("turns", "conversation"):
            if key in parsed:
                return parsed[key]
        raise ValueError(
            f"[{custom_id}] Response dict has no 'turns' or 'conversation' key. "
            f"Keys: {list(parsed.keys())}"
        )
    raise ValueError(
        f"[{custom_id}] Unexpected response type: {type(parsed).__name__}"
    )


def _poll_until_done(client: anthropic.Anthropic, batch_id: str, label: str) -> None:
    """Poll an Anthropic Batch until processing_status == 'ended'.

    Uses exponential backoff up to _POLL_MAX_INTERVAL seconds.
    Raises TimeoutError if the batch has not completed within _POLL_TIMEOUT_SECONDS.
    """
    deadline = time.monotonic() + _POLL_TIMEOUT_SECONDS
    interval = _POLL_INITIAL_INTERVAL

    print(f"Polling {label} batch {batch_id}...")
    while True:
        batch = client.beta.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        total = (
            counts.canceled + counts.errored + counts.expired
            + counts.processing + counts.succeeded
        )
        print(
            f"  Status: {batch.processing_status}  |  "
            f"succeeded={counts.succeeded}  errored={counts.errored}  "
            f"in_progress={counts.processing}  total={total}"
        )
        if batch.processing_status == "ended":
            break

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Batch {batch_id} ({label}) did not complete within "
                f"{_POLL_TIMEOUT_SECONDS // 3600} hours."
            )
        sleep_for = min(interval, remaining, _POLL_MAX_INTERVAL)
        time.sleep(sleep_for)
        interval = min(interval * _POLL_BACKOFF_FACTOR, _POLL_MAX_INTERVAL)


def _build_path_description(path_record: dict, graph_data: dict) -> list[dict]:
    """Build the structured path description sent to the LLM."""
    nodes = graph_data["graph"]["nodes"]
    node_sequence = path_record["node_sequence"]
    edge_labels = path_record["edge_labels"]
    desc = []
    for i, (node_id, label) in enumerate(zip(node_sequence, edge_labels)):
        node_info = nodes.get(node_id, {})
        desc.append({
            "step": i,
            "node_id": node_id,
            "type": node_info.get("type", "unknown"),
            "text": node_info.get("text", ""),
            # label on the incoming edge (what led us here)
            "edge_label_from_prev": label,
            # label on the outgoing edge (which branch we'll take next)
            "edge_label_to_next": edge_labels[i + 1] if i + 1 < len(edge_labels) else None,
        })
    return desc


def submit_pass1_batch(manifest_path: Path | None = None) -> str:
    """Build and submit a single Anthropic batch for all Pass-1 generation calls.

    Returns the batch ID.
    """
    if manifest_path is None:
        manifest_path = PATHS_DIR / "manifest.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest(manifest)
    paths = manifest["paths"]
    graph_cache = _load_graphs()
    _validate_graph_cache(graph_cache, paths)
    client = _get_client()

    requests: list[dict] = []
    for path_record in paths:
        graph_id = path_record["graph_id"]
        path_id = path_record["path_id"]
        graph_data = graph_cache[graph_id]
        path_description = _build_path_description(path_record, graph_data)

        user_content = json.dumps({
            "graph": graph_data["graph"],
            "path": path_description,
            "path_id": path_id,
        }, indent=2)

        requests.append({
            "custom_id": f"pass1__{path_id}",
            "params": {
                "model": _ANTHROPIC_GENERATION_MODEL,
                "max_tokens": 8192,
                "temperature": GENERATION_TEMPERATURE,
                "system": SYSTEM_PROMPT_PASS1,
                "messages": [{"role": "user", "content": user_content}],
            },
        })

    print(f"Submitting Pass-1 batch: {len(requests)} requests...")
    batch = client.beta.messages.batches.create(requests=requests)
    print(f"Batch submitted. ID: {batch.id}")
    print(f"Status: {batch.processing_status}")

    state = _load_state()
    state["pass1_batch_id"] = batch.id
    state["pass1_total"] = len(requests)
    state["pass1_submitted_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _save_state(state)

    return batch.id


def collect_pass1_and_submit_pass2(batch_id: str | None = None) -> str:
    """Poll Pass-1 batch, parse results, and submit the Pass-2 batch.

    Returns the Pass-2 batch ID.
    """
    state = _load_state()
    if batch_id is None:
        batch_id = state.get("pass1_batch_id")
    if not batch_id:
        raise ValueError("No batch ID provided or saved in batch_state.json.")

    manifest = json.loads((PATHS_DIR / "manifest.json").read_text(encoding="utf-8"))
    _validate_manifest(manifest)
    paths_by_id = {p["path_id"]: p for p in manifest["paths"]}
    graph_cache = _load_graphs()
    _validate_graph_cache(graph_cache, manifest["paths"])
    client = _get_client()

    _poll_until_done(client, batch_id, label="Pass-1")

    pass1_results: dict[str, list[dict]] = {}
    for result in client.beta.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            print(f"  WARNING: {result.custom_id} failed — skipping.")
            continue

        content_blocks = result.result.message.content
        if not content_blocks:
            print(f"  WARNING: Empty content for {result.custom_id} — skipping.")
            continue

        path_id = result.custom_id.removeprefix("pass1__")
        try:
            parsed = _parse_json_response(content_blocks[0].text, result.custom_id)
            turns = _parse_turns(parsed, result.custom_id)
            struct_issues = _validate_conversation_structure(turns)
            if struct_issues:
                print(f"  WARNING: {path_id} has structural issues (no retry in "
                      f"batch mode) — {struct_issues}")
            pass1_results[path_id] = turns
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"  WARNING: {exc}")

    # Save Pass-1 results for recovery before submitting Pass-2
    pass1_cache_path = PATHS_DIR.parent / "batch_pass1_results.json"
    pass1_cache_path.write_text(json.dumps(pass1_results, indent=2, ensure_ascii=False))
    print(f"Pass-1 results saved to {pass1_cache_path} ({len(pass1_results)} paths).")

    requests: list[dict] = []
    for path_id, turns in pass1_results.items():
        path_record = paths_by_id.get(path_id)
        if not path_record:
            continue
        graph_data = graph_cache[path_record["graph_id"]]

        verify_content = json.dumps({
            "graph": graph_data["graph"],
            "path": path_record,
            "conversation": turns,
        }, indent=2)
        requests.append({
            "custom_id": f"pass2_verify__{path_id}",
            "params": {
                "model": _ANTHROPIC_VERIFICATION_MODEL,
                "max_tokens": 4096,
                "temperature": VERIFICATION_TEMPERATURE,
                "system": SYSTEM_PROMPT_PASS2,
                "messages": [{"role": "user", "content": verify_content}],
            },
        })

        for key, desc in TRANSFORMATIONS.items():
            system_prompt = ROBUSTNESS_SYSTEM_PROMPT.format(transformation=desc)
            robustness_content = json.dumps({
                "conversation": turns,
                "transformation": key,
            }, indent=2)
            requests.append({
                "custom_id": f"pass2_robust__{path_id}__{key}",
                "params": {
                    "model": _ANTHROPIC_GENERATION_MODEL,
                    "max_tokens": 8192,
                    "temperature": GENERATION_TEMPERATURE,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": robustness_content}],
                },
            })

    if not requests:
        raise RuntimeError(
            "No usable Pass-1 results — all requests failed or could not be parsed. "
            "Check the batch error log before continuing."
        )

    print(f"Submitting Pass-2 batch: {len(requests)} requests...")
    batch2 = client.beta.messages.batches.create(requests=requests)
    print(f"Pass-2 batch submitted. ID: {batch2.id}")

    state["pass2_batch_id"] = batch2.id
    state["pass2_total"] = len(requests)
    state["pass2_submitted_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _save_state(state)

    return batch2.id


def collect_pass2_and_assemble(batch_id: str | None = None) -> list[Path]:
    """Poll Pass-2 batch and write final conversation JSONs to disk.

    Returns list of written file paths.
    """
    state = _load_state()
    if batch_id is None:
        batch_id = state.get("pass2_batch_id")
    if not batch_id:
        raise ValueError("No Pass-2 batch ID provided or saved in batch_state.json.")

    pass1_cache_path = PATHS_DIR.parent / "batch_pass1_results.json"
    if not pass1_cache_path.exists():
        raise FileNotFoundError(
            f"Pass-1 results cache not found at {pass1_cache_path}. "
            "Run the collect-pass1 phase first."
        )
    pass1_results: dict[str, list[dict]] = json.loads(pass1_cache_path.read_text())

    manifest = json.loads((PATHS_DIR / "manifest.json").read_text(encoding="utf-8"))
    _validate_manifest(manifest)
    paths_by_id = {p["path_id"]: p for p in manifest["paths"]}
    graph_cache = _load_graphs()
    _validate_graph_cache(graph_cache, manifest["paths"])
    client = _get_client()
    CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)

    _poll_until_done(client, batch_id, label="Pass-2")

    verifications: dict[str, dict] = {}
    robustness_variants: dict[str, dict[str, list[dict]]] = {}

    for result in client.beta.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            print(f"  WARNING: {result.custom_id} failed — skipping.")
            continue

        content_blocks = result.result.message.content
        if not content_blocks:
            print(f"  WARNING: Empty content for {result.custom_id} — skipping.")
            continue

        custom_id = result.custom_id
        try:
            parsed = _parse_json_response(content_blocks[0].text, custom_id)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"  WARNING: {exc}")
            continue

        if custom_id.startswith("pass2_verify__"):
            path_id = custom_id.removeprefix("pass2_verify__")
            verifications[path_id] = parsed

        elif custom_id.startswith("pass2_robust__"):
            rest = custom_id.removeprefix("pass2_robust__")
            # Format: <path_id>__<transformation_key>
            path_id, key = rest.rsplit("__", 1)
            try:
                turns = _parse_turns(parsed, custom_id)
            except ValueError as exc:
                print(f"  WARNING: {exc}")
                continue
            robustness_variants.setdefault(path_id, {})[key] = turns

    output_files: list[Path] = []
    robustness_keys = list(TRANSFORMATIONS.keys())

    for path_id, base_turns in pass1_results.items():
        path_record = paths_by_id.get(path_id)
        if not path_record:
            continue
        graph_data = graph_cache[path_record["graph_id"]]

        verification_report = verifications.get(path_id, {"verified": True, "issues": []})
        corrected = verification_report.get("corrected_turns")
        corrected_turns = corrected if corrected is not None else base_turns

        base_conv = assemble_conversation(
            path_record=path_record,
            turns=corrected_turns,
            variant="base",
            robustness_axis=None,
            graph_data=graph_data,
            verification_report=verification_report,
        )
        base_path = CONVERSATIONS_DIR / f"{path_id}_base.json"
        base_path.write_text(json.dumps(base_conv, indent=2, ensure_ascii=False), encoding="utf-8")
        output_files.append(base_path)

        for key in robustness_keys:
            variant_turns = robustness_variants.get(path_id, {}).get(key)
            if not variant_turns:
                print(f"  WARNING: Missing robustness variant '{key}' for {path_id} — skipping.")
                continue

            # Robustness variants are not independently verified — they inherit the base verification status.
            variant_verification = {
                "variant_of": "base",
                "base_verified": verification_report.get("verified", True),
                "note": "surface-form variant; not independently verified",
            }
            variant_conv = assemble_conversation(
                path_record=path_record,
                turns=variant_turns,
                variant=key,
                robustness_axis=key,
                graph_data=graph_data,
                verification_report=variant_verification,
            )
            variant_path = CONVERSATIONS_DIR / f"{path_id}_{key}.json"
            variant_path.write_text(
                json.dumps(variant_conv, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            output_files.append(variant_path)

    print(f"\nAssembly complete: {len(output_files)} conversation files written to {CONVERSATIONS_DIR}")
    return output_files
