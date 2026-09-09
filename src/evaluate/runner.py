"""LLM evaluation runner for DiagFlowBench.

Evaluates models across all conversations (clean + mixed). Clean conversations
score on-procedure capability; mixed conversations score off-procedure failure
modes. All model calls go through OpenRouter.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PRINT_LOCK = threading.Lock()

import yaml
import anthropic
from openai import OpenAI

from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    ANTHROPIC_API_KEY,
    EVAL_MODELS,
    EVAL_TEMPERATURE,
    EVAL_MAX_TOKENS,
    CONVERSATIONS_DIR,
    EVALUATION_DIR,
    MODEL_OUTPUTS_DIR,
    SYNTHETIC_GRAPHS_DIR,
    log_api_cost,
    get_model_cost_usd,
)

_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "prompts" / "evaluation.yaml"


def _load_prompts() -> dict[str, Any]:
    return yaml.safe_load(_PROMPTS_PATH.read_text(encoding="utf-8"))


_PROMPTS = _load_prompts()

SYSTEM_PROMPT_TEMPLATE: str = _PROMPTS["system"]


def _get_client(model_id: str = "") -> Any:
    """Create an OpenAI client (OpenRouter) or Anthropic client (native)."""
    if ("anthropic/" in model_id or not OPENROUTER_API_KEY) and ANTHROPIC_API_KEY:
        return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set in .env")
    return OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)


def _build_conversation_messages(
    conversation: dict[str, Any],
    up_to_turn: int,
    system_prompt: str,
) -> list[dict[str, str]]:
    """Build the message history for evaluation up to a given turn."""
    messages = [{"role": "system", "content": system_prompt}]

    turns = conversation.get("turns", [])
    for turn in turns[:up_to_turn]:
        speaker = turn.get("speaker", "")
        utterance = turn.get("utterance", "")

        if speaker == "operator":
            messages.append({"role": "user", "content": utterance})
        elif speaker == "system":
            messages.append({"role": "assistant", "content": utterance})

    return messages


def evaluate_turn(
    client: Any,
    model_id: str,
    messages: list[dict[str, str]],
    *,
    delay: float = 0.5,
    max_tokens: int = EVAL_MAX_TOKENS,
) -> tuple[str, dict[str, int]]:
    """Evaluate a single turn: send messages, get model response.

    Returns (model_response, token_usage). Retries up to 3 times on transient
    API errors (e.g. malformed response body from OpenRouter on large inputs).
    """
    time.sleep(delay)

    last_exc: Exception | None = None
    for attempt in range(3):
        if attempt > 0:
            time.sleep(2 * attempt)  # 2 s, then 4 s
        try:
            if isinstance(client, anthropic.Anthropic):
                # Native Anthropic flow
                clean_model_id = model_id.replace("anthropic/", "")
                if clean_model_id == "claude-haiku-4-5":
                    clean_model_id = "claude-haiku-4-5-20251001"

                system_prompt = ""
                ant_msgs = []
                for m in messages:
                    if m["role"] == "system":
                        system_prompt = m["content"]
                    else:
                        ant_msgs.append(m)

                response = client.messages.create(
                    model=clean_model_id,
                    system=system_prompt,
                    messages=ant_msgs,
                    max_tokens=max_tokens,
                    temperature=EVAL_TEMPERATURE,
                )
                model_response = response.content[0].text
                usage = response.usage
                input_tokens = usage.input_tokens
                output_tokens = usage.output_tokens

                cost = get_model_cost_usd(model_id, input_tokens, output_tokens)

            else:
                # OpenRouter / OpenAI flow
                response = client.chat.completions.create(
                    model=model_id,
                    temperature=EVAL_TEMPERATURE,
                    max_tokens=max_tokens,
                    messages=messages,
                )
                model_response = response.choices[0].message.content or ""
                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0
                cost = get_model_cost_usd(model_id, input_tokens, output_tokens)

            token_usage = {
                "input": input_tokens,
                "output": output_tokens,
            }

            if input_tokens > 0 or output_tokens > 0:
                log_api_cost(
                    model=model_id,
                    phase="evaluation",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_eur=cost,
                )

            return model_response, token_usage

        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                with _PRINT_LOCK:
                    print(f"[{model_id}] Turn retry {attempt + 1}/3: {exc}")

    raise last_exc  # type: ignore[misc]


def evaluate_conversation(
    client: Any,
    model_id: str,
    conversation: dict[str, Any],
    graph_data: dict[str, Any],
    *,
    delay: float = 0.5,
    max_tokens: int = EVAL_MAX_TOKENS,
) -> dict[str, Any]:
    """Evaluate a single conversation, returning a result dict with per-turn model responses."""
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        graph_json=json.dumps(graph_data["graph"], indent=2)
    )

    turns = conversation.get("turns", [])
    operator_turns = [
        (i, t) for i, t in enumerate(turns)
        if t.get("speaker") == "operator"
    ]

    per_turn_results = []
    conversation_messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]

    for turn_idx, (global_idx, turn) in enumerate(operator_turns):
        conversation_messages.append({
            "role": "user",
            "content": turn.get("utterance", ""),
        })

        model_response, token_usage = evaluate_turn(
            client, model_id, conversation_messages, delay=delay, max_tokens=max_tokens
        )

        conversation_messages.append({
            "role": "assistant",
            "content": model_response,
        })

        per_turn_results.append({
            "turn_index": turn_idx,
            "global_turn_index": global_idx,
            "operator_utterance": turn.get("utterance", ""),
            "model_response": model_response,
            "ground_truth": turn.get("ground_truth", {}),
            "edge_label_to_next": turn.get("edge_label_to_next"),
            # Injection metadata (present in mixed conversations only)
            "is_injection": turn.get("is_injection", False),
            "injection_category": turn.get("injection_category"),
            "post_injection": turn.get("post_injection", False),
            "token_usage": token_usage,
        })

    return {
        "conversation_id": conversation.get("conversation_id", ""),
        "graph_id": conversation.get("graph_id", ""),
        "model": model_id,
        "regime": conversation.get("regime", "clean"),  # "clean" or "mixed"
        "num_turns": len(per_turn_results),
        "per_turn": per_turn_results,
        "metadata": {
            "depth": conversation.get("metadata", {}).get("depth"),
            "num_decisions": conversation.get("metadata", {}).get("num_decisions"),
            "terminator_type": conversation.get("metadata", {}).get("terminator_type"),
            "variant": conversation.get("variant"),
            "robustness_axis": conversation.get("robustness_axis"),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _write_results(path: Path, results: list[dict]) -> None:
    """Atomically write results via temp-file rename."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=2), encoding="utf-8")
    tmp.rename(path)


def _evaluate_single_model(
    model_config: dict[str, str],
    conversations: list[dict],
    graph_cache: dict[str, dict],
    output_dir: Path,
    delay: float,
    skip_existing: bool = True,
) -> tuple[str, list[dict]]:
    """Evaluate all conversations for one model. Runs in its own thread.

    Checkpoints every 25 conversations so a crash or credit exhaustion never
    loses more than 25 conversations of work. Re-running the script resumes
    automatically from the last checkpoint.
    """
    model_name  = model_config["name"]
    model_id    = model_config["id"]
    tier        = model_config.get("tier", "unknown")
    max_tokens  = model_config.get("max_tokens", EVAL_MAX_TOKENS)

    out_path = output_dir / f"results_{model_name.replace(' ', '_')}.json"

    results: list[dict] = []
    if out_path.exists():
        results = json.loads(out_path.read_text(encoding="utf-8"))
        if skip_existing and len(results) >= len(conversations):
            with _PRINT_LOCK:
                print(f"[{model_name}] Already complete — skipping ({len(results)} results).")
            return model_name, results

    completed_ids = {r["conversation_id"] for r in results}
    remaining = [c for c in conversations
                 if c.get("conversation_id", "") not in completed_ids]

    client = _get_client(model_id)

    with _PRINT_LOCK:
        done = len(results)
        suffix = f" (resuming from {done})" if done else ""
        print(f"\n[{model_name}] Starting — {len(remaining)} conversations{suffix}.")

    for conv_idx, conv in enumerate(remaining):
        graph_id   = conv.get("graph_id", "unknown")
        graph_data = graph_cache.get(graph_id, {})

        if not graph_data:
            with _PRINT_LOCK:
                print(f"[{model_name}] WARNING: Graph {graph_id} not found — skipping.")
            continue

        conv_id = conv.get("conversation_id", f"conv_{conv_idx}")
        try:
            result = evaluate_conversation(
                client, model_id, conv, graph_data, delay=delay, max_tokens=max_tokens,
            )
            result["tier"] = tier
            results.append(result)
        except Exception as exc:
            with _PRINT_LOCK:
                print(f"[{model_name}] ERROR on {conv_id}: {exc}")
            continue

        if len(results) % 25 == 0:
            _write_results(out_path, results)

        if (conv_idx + 1) % 50 == 0:
            with _PRINT_LOCK:
                print(f"[{model_name}] {conv_idx + 1}/{len(remaining)} done.")

    _write_results(out_path, results)

    with _PRINT_LOCK:
        print(f"[{model_name}] Complete — {len(results)} results → {out_path.name}")

    return model_name, results


def run_evaluation(
    *,
    models: list[dict[str, str]] | None = None,
    conversations_dir: Path | None = None,
    output_dir: Path | None = None,
    delay: float = 0.5,
    max_conversations: int | None = None,
    max_workers: int | None = None,
) -> dict[str, list[dict]]:
    """Run evaluation across all models in parallel.

    Each model is evaluated in its own thread. Conversations within a model
    are evaluated sequentially (the dialogue is stateful). Results are written
    to disk as soon as each model finishes, so the run is fully resumable.
    """
    if models is None:
        models = EVAL_MODELS
    if conversations_dir is None:
        conversations_dir = CONVERSATIONS_DIR
    if output_dir is None:
        output_dir = MODEL_OUTPUTS_DIR
    if max_workers is None:
        max_workers = len(models)
    output_dir.mkdir(parents=True, exist_ok=True)

    conv_files = sorted(conversations_dir.rglob("*.json"))
    if max_conversations:
        conv_files = conv_files[:max_conversations]

    conversations: list[dict] = []
    for f in conv_files:
        if f.name == "quality_report.json":
            continue
        conversations.append(json.loads(f.read_text(encoding="utf-8")))

    graph_cache: dict[str, dict] = {}
    for p in sorted(SYNTHETIC_GRAPHS_DIR.glob("GRAPH*.json")):
        g = json.loads(p.read_text(encoding="utf-8"))
        graph_cache[g["id"]] = g

    print(f"Evaluating {len(models)} models × {len(conversations)} conversations "
          f"with {max_workers} parallel threads.")

    all_results: dict[str, list[dict]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _evaluate_single_model,
                model_config,
                conversations,
                graph_cache,
                output_dir,
                delay,
            ): model_config["name"]
            for model_config in models
        }
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                name, results = future.result()
                all_results[name] = results
            except Exception as exc:
                with _PRINT_LOCK:
                    print(f"[{model_name}] FATAL: {exc}")

    return all_results
