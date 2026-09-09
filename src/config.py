"""Centralised configuration for the DiagFlowBench pipeline.

Loads API keys from .env via python-dotenv and exposes path constants,
budget tracking, and shared settings used across all phases.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent


def _find_env_file(start: Path) -> Path | None:
    """Walk up from start until a .env file is found (handles git worktrees)."""
    current = start
    while True:
        candidate = current / ".env"
        if candidate.exists():
            return candidate
        parent = current.parent
        if parent == current:
            return None
        current = parent


load_dotenv(_find_env_file(ROOT_DIR), override=True)

DATA_DIR = ROOT_DIR / "DiagFlowBench_Dataset"
INPUT_DIR = DATA_DIR / "graphs"
SYNTHETIC_GRAPHS_DIR = INPUT_DIR / "json"

OUTPUT_DIR = DATA_DIR
PATHS_DIR = OUTPUT_DIR / "paths"
CONVERSATIONS_DIR = OUTPUT_DIR / "conversations"
EVALUATION_DIR = ROOT_DIR / "evaluation_results"
MODEL_OUTPUTS_DIR = EVALUATION_DIR / "per_model"  # per-model results_*.json files

DOCS_DIR = ROOT_DIR / "docs"

OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

# Separate key for the LLM-as-judge so judge traffic doesn't compete with
# ongoing evaluations on the main key's rate limits.
JUDGE_API_KEY: str = os.getenv("JUDGE_API_KEY", "")

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

# Note: 'cost_eur' in the spend log is mislabelled — values are in USD.
_SPEND_LOG_PATH = OUTPUT_DIR / ".spend_log.json"
_SPEND_LOCK = threading.Lock()


def _load_spend() -> float:
    """Return cumulative spend from the log file."""
    if _SPEND_LOG_PATH.exists():
        data = json.loads(_SPEND_LOG_PATH.read_text())
        return float(data.get("total_eur", 0.0))
    return 0.0


def _save_spend(total: float, entry: dict | None = None) -> None:
    """Persist updated cumulative spend."""
    _SPEND_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if _SPEND_LOG_PATH.exists():
        data = json.loads(_SPEND_LOG_PATH.read_text())
        existing = data.get("calls", [])
    if entry:
        existing.append(entry)
    _SPEND_LOG_PATH.write_text(
        json.dumps({"total_eur": round(total, 6), "calls": existing}, indent=2)
    )


def log_api_cost(
    *,
    model: str,
    phase: str,
    input_tokens: int,
    output_tokens: int,
    cost_eur: float,
) -> None:
    """Log a single API call cost for documentation.

    No budget enforcement — budget is managed through OpenRouter.
    """
    with _SPEND_LOCK:
        current = _load_spend()
        new_total = current + cost_eur
        _save_spend(
            new_total,
            {
                "model": model,
                "phase": phase,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_eur": round(cost_eur, 6),
            },
        )


def get_total_spend() -> float:
    """Return total spend from the log file."""
    return _load_spend()


GENERATION_MODEL: str = os.getenv(
    "GENERATION_MODEL", "anthropic/claude-sonnet-4-5"
)
GENERATION_TEMPERATURE: float = float(os.getenv("GENERATION_TEMPERATURE", "0.7"))

VERIFICATION_MODEL: str = os.getenv("VERIFICATION_MODEL", GENERATION_MODEL)
VERIFICATION_TEMPERATURE: float = 0.0

# Reasoning models use max_tokens=2048 to accommodate chain-of-thought scratchpad.
# Panel of ten models (paper Appendix, Table: Models evaluated, by category).
EVAL_MODELS = [
    {"name": "Gemini 2.5 Flash",     "id": "google/gemini-2.5-flash",                  "tier": "commercial"},
    {"name": "GPT-4o Mini",          "id": "openai/gpt-4o-mini",                        "tier": "commercial"},
    {"name": "Qwen3 235B Thinking",  "id": "qwen/qwen3-235b-a22b-thinking-2507",        "tier": "open-weight", "max_tokens": 2048},
    {"name": "Qwen3 30B Thinking",   "id": "qwen/qwen3-30b-a3b-thinking-2507",          "tier": "open-weight", "max_tokens": 2048},
    {"name": "Llama 3.3 70B",        "id": "meta-llama/llama-3.3-70b-instruct",         "tier": "open-weight"},
    {"name": "Llama 4 Scout",        "id": "meta-llama/llama-4-scout",                  "tier": "open-weight"},
    {"name": "Llama 4 Maverick",     "id": "meta-llama/llama-4-maverick",               "tier": "open-weight"},
    {"name": "Mistral Small 24B",    "id": "mistralai/mistral-small-2603",               "tier": "open-weight"},
    {"name": "GPT-OSS 120B",         "id": "openai/gpt-oss-120b",                       "tier": "open-weight"},
    {"name": "Nemotron 3 Super",      "id": "nvidia/nemotron-3-super-120b-a12b",          "tier": "open-weight"},
]

EVAL_TEMPERATURE: float = 0.0
EVAL_MAX_TOKENS: int = 512

GRAPH_SIZE_SMALL_MAX: int = 14   # ≤14 nodes
GRAPH_SIZE_MEDIUM_MAX: int = 30  # 15–30 nodes
# > 30 nodes = large

# Must match the number of entries under 'transformations' in generation.yaml.
ROBUSTNESS_VARIANTS: int = 1

# Pricing in USD per 1M tokens. The spend log field is labelled 'cost_eur' for historical reasons but stores USD.
_MODEL_PRICING_USD_PER_1M: dict[str, dict[str, float]] = {
    "claude-sonnet-4-5":          {"input": 3.00,  "output": 15.00},
    "claude-opus-4-7":            {"input": 15.00, "output": 75.00},
    "claude-haiku-4-5":           {"input": 0.80,  "output": 4.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00,  "output": 15.00},
    "claude-3-5-haiku-20241022":  {"input": 0.80,  "output": 4.00},
    "claude-3-haiku-20240307":    {"input": 0.25,  "output": 1.25},
}
_DEFAULT_PRICING_USD_PER_1M: dict[str, float] = {"input": 3.00, "output": 15.00}


def get_model_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return API cost in USD for a given model and token counts.

    Strips any provider prefix (e.g. 'anthropic/') before looking up pricing.
    Falls back to claude-sonnet-4 rates for unknown models.
    """
    bare = model.split("/")[-1]
    pricing = _MODEL_PRICING_USD_PER_1M.get(bare, _DEFAULT_PRICING_USD_PER_1M)
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

MAX_LOOP_ITERATIONS: int = 2

# Per-conversation injection rate sampled from Uniform[MIN, MAX]; k = max(1, round(candidates * rate)).
INJECTION_RATE_MIN: float = 0.05
INJECTION_RATE_MAX: float = 0.30

SELF_BLEU_THRESHOLD: float = 0.3
EDGE_LABEL_CONSISTENCY_THRESHOLD: float = 0.85

JACCARD_THRESHOLD_DEFAULT: float = 0.05
JACCARD_SWEEP_THRESHOLDS: list[float] = [round(i * 0.05, 2) for i in range(1, 11)]

TOTAL_GRAPHS: int = 50
