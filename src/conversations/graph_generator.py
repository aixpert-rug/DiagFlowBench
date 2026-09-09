"""Conversation generation for DiagFlowBench.

Two-pass pipeline: Pass 1 generates scripted operator utterances for each path;
Pass 2 verifies and corrects each utterance for edge-label consistency and step leakage.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

from src.config import (
    ANTHROPIC_API_KEY,
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
    VERIFICATION_MODEL,
    VERIFICATION_TEMPERATURE,
    SYNTHETIC_GRAPHS_DIR,
    PATHS_DIR,
    CONVERSATIONS_DIR,
    get_model_cost_usd,
    log_api_cost,
)

_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "prompts" / "generation.yaml"

_REQUIRED_PROMPT_KEYS = {"pass1_system", "pass2_system", "robustness_system", "transformations"}


def _load_prompts() -> dict[str, Any]:
    prompts = yaml.safe_load(_PROMPTS_PATH.read_text(encoding="utf-8"))
    missing = _REQUIRED_PROMPT_KEYS - set(prompts.keys())
    if missing:
        raise ValueError(
            f"generation.yaml is missing required keys: {missing}. "
            f"Found: {set(prompts.keys())}"
        )
    if not isinstance(prompts["transformations"], dict) or not prompts["transformations"]:
        raise ValueError("generation.yaml 'transformations' must be a non-empty dict")
    return prompts


_PROMPTS = _load_prompts()

SYSTEM_PROMPT_PASS1: str = _PROMPTS["pass1_system"]
SYSTEM_PROMPT_PASS2: str = _PROMPTS["pass2_system"]
ROBUSTNESS_SYSTEM_PROMPT: str = _PROMPTS["robustness_system"]
TRANSFORMATIONS: dict[str, str] = {k: v.strip() for k, v in _PROMPTS["transformations"].items()}

# Strip any provider prefix (e.g. "anthropic/") — the Anthropic API only accepts
# bare model IDs. The prefix is an OpenRouter convention used in config defaults.
_API_GENERATION_MODEL: str = GENERATION_MODEL.split("/")[-1]
_API_VERIFICATION_MODEL: str = VERIFICATION_MODEL.split("/")[-1]


def _get_client() -> OpenAI:
    """Create an OpenAI client configured for the direct Anthropic API."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set in .env. Generation requires a direct "
            "Anthropic API key. OpenRouter is only used for evaluation."
        )
    return OpenAI(
        api_key=ANTHROPIC_API_KEY,
        base_url="https://api.anthropic.com/v1/",
    )


def _parse_json_text(text: str, phase: str) -> Any:
    """Parse JSON from an LLM response, with fallback for code-fenced output.

    The Anthropic API's OpenAI-compatible endpoint does not support
    response_format enforcement, so models may wrap JSON in markdown fences.
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
        f"[{phase}] Could not parse JSON from response. "
        f"Preview: {text[:300]!r}"
    )


def _parse_turns_response(result: Any, phase: str) -> list[dict[str, Any]]:
    """Extract a turn list from a parsed LLM JSON response.

    Raises ValueError on unexpected structure rather than silently falling back,
    so caller retry logic can act on the failure.
    """
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        for key in ("turns", "conversation"):
            if key in result:
                return result[key]
        raise ValueError(
            f"[{phase}] LLM returned a dict with no 'turns' or 'conversation' key. "
            f"Keys present: {list(result.keys())}"
        )
    raise ValueError(
        f"[{phase}] Unexpected LLM response type: {type(result).__name__}"
    )


def _call_llm(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    phase: str,
    max_retries: int = 3,
    delay: float = 1.0,
) -> dict[str, Any]:
    """Make an LLM API call with retry logic and cost logging.

    Returns the parsed JSON response.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            usage = response.usage
            if usage:
                cost = get_model_cost_usd(model, usage.prompt_tokens, usage.completion_tokens)
                log_api_cost(
                    model=model,
                    phase=phase,
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    cost_eur=cost,
                )

            content = response.choices[0].message.content
            return _parse_json_text(content, phase)

        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))
                continue
            raise
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))
                continue
            raise


def _validate_conversation_structure(turns: list[dict[str, Any]]) -> list[str]:
    """Rule-based structural check — no LLM needed.

    Verifies that speakers strictly alternate (system → operator → system → …)
    with no consecutive same-speaker turns. Returns a list of human-readable
    issue strings; empty list means the structure is clean.

    Called after Pass 1 so structural problems trigger a retry before the
    (token-expensive) Pass 2 verifier is invoked.
    """
    issues: list[str] = []

    if not turns:
        issues.append("Empty turn list")
        return issues

    if turns[0].get("speaker") != "system":
        issues.append(
            f"Turn 0: first turn must be 'system', got {turns[0].get('speaker')!r}"
        )

    prev_speaker: str | None = None
    for i, turn in enumerate(turns):
        speaker = turn.get("speaker")
        if speaker not in ("system", "operator"):
            issues.append(f"Turn {i}: unknown speaker {speaker!r}")
        # Consecutive system turns are expected: the START TERMINATOR produces
        # a single system turn, so the next node's system instruction naturally
        # follows it. Only consecutive OPERATOR turns are a real problem —
        # that means a process/decision instruction is missing before the response.
        if speaker == "operator" and prev_speaker == "operator":
            issues.append(
                f"Turn {i}: consecutive 'operator' turns — missing system turn "
                f"before node {turn.get('node_id', '?')}"
            )
        prev_speaker = speaker

    return issues


def _validate_inputs(
    graph_cache: dict[str, dict],
    paths: list[dict[str, Any]],
) -> None:
    """Validate that every path's graph_id is present in the loaded graph cache.

    Raises ValueError with a list of missing graph IDs so the operator can fix
    the dataset before any API calls are made.
    """
    missing = sorted({p["graph_id"] for p in paths} - set(graph_cache))
    if missing:
        raise ValueError(
            f"The following graph IDs are referenced in the manifest but not "
            f"found in {SYNTHETIC_GRAPHS_DIR}:\n  " + "\n  ".join(missing)
        )


def _validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate that the manifest has the expected structure.

    Checks for required top-level keys and that each path record has the fields
    needed by the generation pipeline.
    """
    if "paths" not in manifest:
        raise ValueError("Manifest is missing the 'paths' key.")
    required_path_fields = {
        "path_id", "graph_id", "node_sequence", "edge_labels", "decision_points",
        "depth", "num_decisions", "num_processes", "num_documents",
        "terminator_type", "terminator_text",
    }
    for i, path in enumerate(manifest["paths"]):
        missing = required_path_fields - set(path.keys())
        if missing:
            raise ValueError(
                f"Path record at index {i} (path_id={path.get('path_id', '?')!r}) "
                f"is missing required fields: {missing}"
            )


def generate_conversation_pass1(
    client: OpenAI,
    graph_data: dict[str, Any],
    path_record: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate operator utterances for a single path (Pass 1)."""
    nodes = graph_data["graph"]["nodes"]
    node_sequence = path_record["node_sequence"]
    edge_labels = path_record["edge_labels"]

    path_description = []
    for i, (node_id, label) in enumerate(zip(node_sequence, edge_labels)):
        node_info = nodes.get(node_id, {})
        entry = {
            "step": i,
            "node_id": node_id,
            "type": node_info.get("type", "unknown"),
            "text": node_info.get("text", ""),
            # label on the incoming edge (what led us here)
            "edge_label_from_prev": label,
            # label on the outgoing edge (which branch we'll take next)
            "edge_label_to_next": edge_labels[i + 1] if i + 1 < len(edge_labels) else None,
        }
        path_description.append(entry)

    user_prompt = json.dumps({
        "graph": graph_data["graph"],
        "path": path_description,
        "path_id": path_record["path_id"],
    }, indent=2)

    result = _call_llm(
        client,
        model=_API_GENERATION_MODEL,
        system_prompt=SYSTEM_PROMPT_PASS1,
        user_prompt=user_prompt,
        temperature=GENERATION_TEMPERATURE,
        phase="generation_pass1",
    )

    return _parse_turns_response(result, "generation_pass1")


def verify_conversation(
    client: OpenAI,
    graph_data: dict[str, Any],
    path_record: dict[str, Any],
    turns: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify and correct a generated conversation (Pass 2).

    Returns (corrected_turns, verification_report).
    """
    user_prompt = json.dumps({
        "graph": graph_data["graph"],
        "path": path_record,
        "conversation": turns,
    }, indent=2)

    result = _call_llm(
        client,
        model=_API_VERIFICATION_MODEL,
        system_prompt=SYSTEM_PROMPT_PASS2,
        user_prompt=user_prompt,
        temperature=VERIFICATION_TEMPERATURE,
        phase="generation_pass2",
    )

    if result.get("verified", True) and not result.get("issues"):
        return turns, {"verified": True, "issues": []}

    corrected = result.get("corrected_turns", turns)
    return corrected, result


def generate_robustness_variant(
    client: OpenAI,
    base_turns: list[dict[str, Any]],
    transformation_key: str,
) -> list[dict[str, Any]]:
    """Generate a robustness variant of a conversation."""
    transformation_desc = TRANSFORMATIONS[transformation_key]
    system_prompt = ROBUSTNESS_SYSTEM_PROMPT.format(transformation=transformation_desc)

    user_prompt = json.dumps({
        "conversation": base_turns,
        "transformation": transformation_key,
    }, indent=2)

    result = _call_llm(
        client,
        model=_API_GENERATION_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=GENERATION_TEMPERATURE,
        phase="robustness_expansion",
    )

    return _parse_turns_response(result, f"robustness_expansion[{transformation_key}]")


def assemble_conversation(
    *,
    path_record: dict[str, Any],
    turns: list[dict[str, Any]],
    variant: str,
    robustness_axis: str | None,
    graph_data: dict[str, Any],
    verification_report: dict[str, Any],
) -> dict[str, Any]:
    """Assemble a complete conversation object with metadata."""
    nodes = graph_data["graph"]["nodes"]
    node_sequence = path_record["node_sequence"]

    annotated_turns = []
    for i, turn in enumerate(turns):
        annotated = dict(turn)
        annotated["turn_index"] = i

        if "node_id" in turn:
            nid = turn["node_id"]
            node_info = nodes.get(nid, {})
            annotated["ground_truth"] = {
                "node_id": nid,
                "node_type": node_info.get("type", "unknown"),
                "node_text": node_info.get("text", ""),
            }

        annotated_turns.append(annotated)

    return {
        "conversation_id": f"{path_record['path_id']}_{variant}",
        "graph_id": path_record["graph_id"],
        "path_id": path_record["path_id"],
        "regime": "clean",
        "variant": variant,
        "robustness_axis": robustness_axis,
        "turns": annotated_turns,
        "metadata": {
            "depth": path_record["depth"],
            "num_decisions": path_record["num_decisions"],
            "num_processes": path_record["num_processes"],
            "num_documents": path_record["num_documents"],
            "terminator_type": path_record["terminator_type"],
            "terminator_text": path_record["terminator_text"],
            # true_terminator=False means the path ends at a dead-end process/decision
            # node (no explicit terminator). Termination recognition is NOT scored
            # for these conversations. See paper §5.3.
            "true_terminator": path_record.get("true_terminator", True),
            "node_sequence": node_sequence,
            "edge_labels": path_record["edge_labels"],
            "decision_points": path_record["decision_points"],
            "generation_model": GENERATION_MODEL,
            "verification": verification_report,
        },
    }


def run_generation_pipeline(
    *,
    manifest_path: Path | None = None,
    output_dir: Path | None = None,
    skip_existing: bool = True,
    delay_between_calls: float = 0.5,
) -> list[Path]:
    """Run the full two-pass generation pipeline for all paths."""
    if manifest_path is None:
        manifest_path = PATHS_DIR / "manifest.json"
    if output_dir is None:
        output_dir = CONVERSATIONS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest(manifest)
    paths = manifest["paths"]
    client = _get_client()

    graph_cache: dict[str, dict] = {}
    for p in sorted(SYNTHETIC_GRAPHS_DIR.glob("GRAPH*.json")):
        g = json.loads(p.read_text(encoding="utf-8"))
        graph_cache[g["id"]] = g
    _validate_inputs(graph_cache, paths)

    output_files: list[Path] = []
    total = len(paths)
    robustness_keys = list(TRANSFORMATIONS.keys())

    for idx, path_record in enumerate(paths):
        graph_id = path_record["graph_id"]
        path_id = path_record["path_id"]
        graph_data = graph_cache[graph_id]

        print(f"[{idx + 1}/{total}] Generating {path_id}...")

        base_path = output_dir / f"{path_id}_base.json"
        if skip_existing and base_path.exists():
            print(f"  Skipping (already exists)")
            output_files.append(base_path)
            for key in robustness_keys:
                variant_path = output_dir / f"{path_id}_{key}.json"
                if variant_path.exists():
                    output_files.append(variant_path)
            continue

        # Pass 1 — generate, with structural retry before calling the LLM verifier.
        _STRUCT_RETRIES = 2
        turns: list[dict[str, Any]] = []
        struct_issues: list[str] = []
        for attempt in range(1, _STRUCT_RETRIES + 2):
            try:
                turns = generate_conversation_pass1(client, graph_data, path_record)
                time.sleep(delay_between_calls)
            except Exception as e:
                print(f"  ERROR in Pass 1 (attempt {attempt}): {e}")
                if attempt <= _STRUCT_RETRIES:
                    continue
                break

            struct_issues = _validate_conversation_structure(turns)
            if not struct_issues:
                break
            print(f"  Structural issues on attempt {attempt}: {struct_issues}")
            if attempt <= _STRUCT_RETRIES:
                print(f"  Retrying Pass 1...")

        if not turns:
            print(f"  Skipping {path_id} — Pass 1 failed entirely.")
            continue
        if struct_issues:
            print(f"  WARNING: Structural issues persist after {_STRUCT_RETRIES + 1} "
                  f"attempts — proceeding anyway: {struct_issues}")

        try:
            corrected_turns, verification_report = verify_conversation(
                client, graph_data, path_record, turns
            )
            time.sleep(delay_between_calls)
        except Exception as e:
            print(f"  WARNING: Verification failed ({e}), using uncorrected turns")
            corrected_turns = turns
            verification_report = {"verified": False, "error": str(e)}

        base_conv = assemble_conversation(
            path_record=path_record,
            turns=corrected_turns,
            variant="base",
            robustness_axis=None,
            graph_data=graph_data,
            verification_report=verification_report,
        )
        base_path.write_text(
            json.dumps(base_conv, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        output_files.append(base_path)

        for key in robustness_keys:
            variant_path = output_dir / f"{path_id}_{key}.json"
            try:
                variant_turns = generate_robustness_variant(client, corrected_turns, key)
                time.sleep(delay_between_calls)

                # Robustness variants are not independently verified — they inherit
                # the base verification status with a note.
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
                variant_path.write_text(
                    json.dumps(variant_conv, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                output_files.append(variant_path)
            except Exception as e:
                print(f"  WARNING: Robustness variant '{key}' failed: {e}")

    print(f"\nGeneration complete: {len(output_files)} conversations written")
    return output_files
