"""LangChain ChatOpenAI wrapper with OpenRouter/local backend support."""
import os
from urllib import error as urllib_error
from urllib import request as urllib_request
import json

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_DEFAULT_BASE = "https://openrouter.ai/api/v1"
OLLAMA_DEFAULT_BASE = "http://127.0.0.1:11434"
LLAMACPP_DEFAULT_BASE = "http://127.0.0.1:11439"
OLLAMA_PREFIX = "ollama:"
LLAMACPP_PREFIX = "llamacpp:"


def is_ollama_model(model_id: str) -> bool:
    """Return True if model is in SlovakBench Ollama notation."""
    return model_id.startswith(OLLAMA_PREFIX)


def is_llamacpp_model(model_id: str) -> bool:
    """Return True if model is in SlovakBench llama.cpp notation."""
    return model_id.startswith(LLAMACPP_PREFIX)


def is_local_model(model_id: str) -> bool:
    """Return True if model uses a local OpenAI-compatible backend."""
    return is_ollama_model(model_id) or is_llamacpp_model(model_id)


def normalize_model_name(model_id: str) -> str:
    """Strip SlovakBench local-provider prefix if present."""
    if is_ollama_model(model_id):
        return model_id[len(OLLAMA_PREFIX):]
    if is_llamacpp_model(model_id):
        return model_id[len(LLAMACPP_PREFIX):]
    return model_id


def _normalize_base_url(url: str) -> str:
    return url.rstrip("/")


def _resolve_backend(model_id: str, backend: str | None = None) -> str:
    if backend:
        resolved = backend.strip().lower()
    elif is_ollama_model(model_id):
        resolved = "ollama"
    elif is_llamacpp_model(model_id):
        resolved = "llamacpp"
    else:
        resolved = os.getenv("LLM_BACKEND", "openrouter").strip().lower()

    if resolved not in {"openrouter", "ollama", "llamacpp"}:
        raise ValueError(
            f"Unsupported LLM_BACKEND='{resolved}'. Use 'openrouter', 'ollama', or 'llamacpp'."
        )
    return resolved


def get_ollama_base_url() -> str:
    """Return Ollama base URL without trailing slash."""
    return _normalize_base_url(os.getenv("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE))


def get_ollama_openai_base() -> str:
    """Return Ollama OpenAI-compatible endpoint."""
    return f"{get_ollama_base_url()}/v1"


def get_llamacpp_base_url() -> str:
    """Return llama.cpp server base URL without trailing slash."""
    return _normalize_base_url(os.getenv("LLAMACPP_BASE_URL", LLAMACPP_DEFAULT_BASE))


def get_llamacpp_openai_base() -> str:
    """Return llama.cpp OpenAI-compatible endpoint."""
    return f"{get_llamacpp_base_url()}/v1"


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


def list_openai_compatible_models(
    base_url: str, timeout_sec: float = 5.0, api_key: str | None = None
) -> list[str]:
    """Fetch model names from an OpenAI-compatible /v1/models endpoint."""
    models_url = f"{_normalize_base_url(base_url)}/v1/models"
    request = urllib_request.Request(models_url)
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib_request.urlopen(request, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib_error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not reach OpenAI-compatible models endpoint at {models_url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from OpenAI-compatible endpoint at {models_url}") from exc

    models = payload.get("data", [])
    names = sorted({m.get("id", "").strip() for m in models if m.get("id")})
    return names


def list_llamacpp_models(timeout_sec: float = 5.0) -> list[str]:
    """Fetch available llama.cpp model names from /v1/models."""
    api_key = os.getenv("LLAMACPP_API_KEY") or os.getenv("OPENAI_API_KEY")
    return list_openai_compatible_models(
        get_llamacpp_base_url(), timeout_sec=timeout_sec, api_key=api_key
    )


def create_llm(
    model_id: str,
    extra_body: dict | None = None,
    temperature: float = 1.0,
    backend: str | None = None,
) -> ChatOpenAI:
    """Create ChatOpenAI configured for OpenRouter or a local provider."""
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

    if resolved_backend == "llamacpp":
        return ChatOpenAI(
            model=normalized_model,
            openai_api_key=os.getenv("LLAMACPP_API_KEY")
            or os.getenv("OPENAI_API_KEY", "llamacpp"),
            openai_api_base=get_llamacpp_openai_base(),
            temperature=temperature,
            **(
                {"max_tokens": int(os.getenv("LLAMACPP_MAX_TOKENS"))}
                if os.getenv("LLAMACPP_MAX_TOKENS")
                else {}
            ),
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
