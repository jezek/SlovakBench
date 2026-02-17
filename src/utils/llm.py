"""LangChain ChatOpenAI wrapper with OpenRouter/Ollama backend support."""
import os
from urllib import error as urllib_error
from urllib import request as urllib_request
import json

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_DEFAULT_BASE = "https://openrouter.ai/api/v1"
OLLAMA_DEFAULT_BASE = "http://127.0.0.1:11434"
OLLAMA_PREFIX = "ollama:"


def is_ollama_model(model_id: str) -> bool:
    """Return True if model is in SlovakBench Ollama notation."""
    return model_id.startswith(OLLAMA_PREFIX)


def normalize_model_name(model_id: str) -> str:
    """Strip SlovakBench Ollama prefix if present."""
    if is_ollama_model(model_id):
        return model_id[len(OLLAMA_PREFIX):]
    return model_id


def _normalize_base_url(url: str) -> str:
    return url.rstrip("/")


def _resolve_backend(model_id: str, backend: str | None = None) -> str:
    if backend:
        resolved = backend.strip().lower()
    elif is_ollama_model(model_id):
        resolved = "ollama"
    else:
        resolved = os.getenv("LLM_BACKEND", "openrouter").strip().lower()

    if resolved not in {"openrouter", "ollama"}:
        raise ValueError(f"Unsupported LLM_BACKEND='{resolved}'. Use 'openrouter' or 'ollama'.")
    return resolved


def get_ollama_base_url() -> str:
    """Return Ollama base URL without trailing slash."""
    return _normalize_base_url(os.getenv("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE))


def get_ollama_openai_base() -> str:
    """Return Ollama OpenAI-compatible endpoint."""
    return f"{get_ollama_base_url()}/v1"


def list_ollama_models(timeout_sec: float = 5.0) -> list[str]:
    """Fetch available Ollama model names from /api/tags."""
    tags_url = f"{get_ollama_base_url()}/api/tags"
    try:
        with urllib_request.urlopen(tags_url, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib_error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not reach Ollama at {tags_url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from Ollama at {tags_url}") from exc

    models = payload.get("models", [])
    names = sorted({m.get("name", "").strip() for m in models if m.get("name")})
    return names


def create_llm(
    model_id: str,
    extra_body: dict | None = None,
    temperature: float = 1.0,
    backend: str | None = None,
) -> ChatOpenAI:
    """Create ChatOpenAI configured for OpenRouter or Ollama."""
    resolved_backend = _resolve_backend(model_id, backend)
    normalized_model = normalize_model_name(model_id).strip()
    if not normalized_model:
        raise ValueError("Model name cannot be empty.")

    if resolved_backend == "ollama":
        return ChatOpenAI(
            model=normalized_model,
            openai_api_key=os.getenv("OPENAI_API_KEY", "ollama"),
            openai_api_base=get_ollama_openai_base(),
            temperature=temperature,
        )

    body = {"usage": {"include": True}}

    if extra_body:
        extra_payload = dict(extra_body)
        if "reasoning" in extra_payload and isinstance(extra_payload["reasoning"], dict):
            # OpenRouter supports stripping reasoning traces from output.
            extra_payload["reasoning"] = dict(extra_payload["reasoning"])
            extra_payload["reasoning"]["exclude"] = True
        body.update(extra_payload)

    return ChatOpenAI(
        model=normalized_model,
        openai_api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or "openrouter",
        openai_api_base=os.getenv("OPENAI_API_BASE", OPENROUTER_DEFAULT_BASE),
        temperature=temperature,
        extra_body=body,
    )


def get_cost(response) -> float:
    """Extract cost in USD from LangChain response metadata."""
    if not response or not hasattr(response, "response_metadata"):
        return 0.0
    metadata = response.response_metadata or {}
    token_usage = metadata.get("token_usage", {})
    return token_usage.get("cost", 0.0) or 0.0


def print_cost(response, label: str = ""):
    """Print cost from LangChain response."""
    cost = get_cost(response)
    prefix = f"{label}: " if label else ""
    print(f"💰 {prefix}${cost:.6f}")
