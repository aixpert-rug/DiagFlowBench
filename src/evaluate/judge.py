"""LLM-as-judge for off-procedure injection turn classification.

Uses claude-haiku-4.5 (via OpenRouter) as the judge model. Results are
persisted in a local cache file so re-running analysis never makes redundant
API calls.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import openai
import yaml

from src.config import JUDGE_API_KEY, OPENROUTER_BASE_URL, EVALUATION_DIR

JUDGE_MODEL = "anthropic/claude-haiku-4.5"
JUDGE_CACHE_PATH = EVALUATION_DIR / "judge_cache.json"

_MAX_RETRIES = 3
_RETRY_DELAYS = [2.0, 4.0, 8.0]

_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "prompts" / "evaluation.yaml"

def _load_prompts() -> dict:
    return yaml.safe_load(_PROMPTS_PATH.read_text(encoding="utf-8"))

_PROMPTS = _load_prompts()


def _load_cache() -> dict[str, Any]:
    if JUDGE_CACHE_PATH.exists():
        try:
            return json.loads(JUDGE_CACHE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict[str, Any]) -> None:
    JUDGE_CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _cache_key(task: str, model_name: str, conv_id: str, turn_index: int) -> str:
    return f"{task}|{model_name}|{conv_id}|{turn_index}"


def _call_judge(prompt: str, max_tokens: int = 20) -> str:
    """Call Haiku via OpenRouter with retries; return the stripped response text."""
    client = openai.OpenAI(api_key=JUDGE_API_KEY, base_url=OPENROUTER_BASE_URL)
    last_err: Exception | None = None
    for attempt, delay in enumerate([0.0] + _RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                max_tokens=max_tokens,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()
        except openai.APIError as exc:
            last_err = exc
    raise RuntimeError(f"Judge API failed after {_MAX_RETRIES} retries: {last_err}")



def _graph_summary(graph_nodes: dict[str, Any]) -> str:
    """Produce the full node-list string for the judge prompt.

    The judge prompt states these are the ONLY valid diagnostic actions the
    assistant may suggest, so every node must be listed; graphs run up to 60
    nodes and a partial list would misclassify a genuine forced-mapping onto
    an omitted node as fabrication.
    """
    lines = []
    for nid, node in graph_nodes.items():
        text = node.get("text", "").replace("\n", " ").strip()
        ntype = node.get("type", "process")
        lines.append(f"  {nid} ({ntype}): {text}")
    return "\n".join(lines)



_OUTCOME_MAP = {
    "CORRECT_ABSTENTION": "correct_abstention",
    "FORCED_MAPPING":     "forced_mapping",
    "FABRICATION":        "fabrication",
}


def classify_injection_turn(
    *,
    model_response: str,
    operator_utterance: str,
    graph_nodes: dict[str, Any],
    model_name: str,
    conv_id: str,
    turn_index: int,
    cache: dict[str, Any],
) -> dict[str, Any]:
    """Classify one injection turn; reads/writes cache in-place.

    cache is a shared mutable dict — caller loads once, passes through, saves when done.
    """
    key = _cache_key("inj", model_name, conv_id, turn_index)
    if key in cache:
        return cache[key]

    prompt = _PROMPTS["judge_injection"].format(
        graph_summary=_graph_summary(graph_nodes),
        utterance=operator_utterance[:500],
        response=model_response[:600],
    )

    raw = _call_judge(prompt, max_tokens=20)
    # Normalise: strip punctuation, uppercase, partial matches
    raw_upper = raw.upper().split()[0] if raw.split() else ""
    outcome_key = None
    for label in _OUTCOME_MAP:
        if label in raw_upper or raw_upper in label:
            outcome_key = label
            break
    if outcome_key is None:
        # Fallback: default to fabrication if judge output is unrecognised
        outcome_key = "FABRICATION"

    outcome = _OUTCOME_MAP[outcome_key]
    result = {
        "outcome":            outcome,
        "fabrication":        outcome == "fabrication",
        "forced_mapping":     outcome == "forced_mapping",
        "correct_abstention": outcome == "correct_abstention",
        "judge_raw":          raw,
    }
    cache[key] = result
    return result



def classify_post_injection_turn(
    *,
    model_response: str,
    operator_utterance: str,
    expected_node_text: str,
    model_name: str,
    conv_id: str,
    turn_index: int,
    cache: dict[str, Any],
) -> dict[str, Any]:
    """Classify whether a post-injection on-procedure turn shows inappropriate abstention."""
    key = _cache_key("post", model_name, conv_id, turn_index)
    if key in cache:
        return cache[key]

    prompt = _PROMPTS["judge_post_injection"].format(
        expected_node_text=expected_node_text[:300],
        utterance=operator_utterance[:400],
        response=model_response[:500],
    )

    raw = _call_judge(prompt, max_tokens=5)
    is_ia = raw.upper().startswith("YES")
    result = {
        "inappropriate_abstention": is_ia,
        "judge_raw": raw,
    }
    cache[key] = result
    return result
