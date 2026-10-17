"""
============================================================
CineMind — Centralized Groq LLM provider
============================================================

Only the CHATBOT uses this module. It wraps the official Groq
Python SDK (the `groq` package) so that client creation,
configuration, model selection and error handling live in exactly
one place.

Design rules (from the project spec):
    * Groq is the ONLY external AI provider for the chatbot.
      This module never silently switches to another provider.
    * The API key is read from the environment and is NEVER logged,
      printed or returned in error messages.
    * If Groq is not configured or unreachable, callers raise a
      clear, catchable error so the chatbot can use its local
      deterministic fallback.

Environment variables
---------------------
    GROQ_API_KEY   (required)  API key for Groq.
                   `AI_API_KEY` is accepted as a deprecated fallback.
    GROQ_MODEL     (optional)  Chat model id.
                   Default: "openai/gpt-oss-120b" (a current
                   Groq production model).
                   `AI_MODEL` is accepted as a deprecated fallback.
    GROQ_BASE_URL  (optional)  Override the Groq API base URL.
                   Defaults to the SDK's own Groq endpoint.
    GROQ_TIMEOUT   (optional)  Per-request timeout in seconds.
                   Default: 30.
"""

import os
import threading

try:  # The SDK is optional at import time so the rest of CineMind
    # (search, recommendations, watchlist) keeps working without it.
    import groq as groq_sdk
    from groq import Groq
    _GROQ_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on install state
    groq_sdk = None
    Groq = None
    _GROQ_IMPORT_ERROR = exc


# Model verified against Groq's official model list:
#   https://console.groq.com/docs/models  ("Production Models")
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_GROQ_TIMEOUT = 30.0


class LLMError(RuntimeError):
    """Base class for chatbot LLM failures (safe to show to operators)."""


class LLMNotConfigured(LLMError):
    """No API key is configured, so the chatbot must use its local fallback."""


class LLMUnavailable(LLMError):
    """Groq was configured but the request failed (auth, rate limit, network)."""


_client = None
_client_lock = threading.Lock()
_deprecation_warned = False


def _clean(value):
    return (value or "").strip()


def get_api_key() -> str:
    """Return the Groq API key, preferring GROQ_API_KEY.

    Falls back to the legacy AI_API_KEY for backward compatibility and
    prints a one-time migration warning (never the key itself).
    """
    global _deprecation_warned
    key = _clean(os.getenv("GROQ_API_KEY"))
    if key:
        return key
    legacy = _clean(os.getenv("AI_API_KEY"))
    if legacy:
        if not _deprecation_warned:
            _deprecation_warned = True
            print(
                "[CineMind] NOTE: AI_API_KEY is deprecated for the chatbot; "
                "rename it to GROQ_API_KEY in your local .env. Using it for now.",
            )
        return legacy
    return ""


def get_model() -> str:
    """Return the configured Groq model id (never empty)."""
    model = _clean(os.getenv("GROQ_MODEL"))
    if model:
        return model
    legacy = _clean(os.getenv("AI_MODEL"))
    if legacy:
        return legacy
    return DEFAULT_GROQ_MODEL


def get_timeout() -> float:
    """Return the request timeout in seconds (default 30)."""
    raw = _clean(os.getenv("GROQ_TIMEOUT"))
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            print(
                f"[CineMind] WARNING: GROQ_TIMEOUT='{raw}' is not a number; "
                f"using {DEFAULT_GROQ_TIMEOUT:.0f}s.",
            )
    return DEFAULT_GROQ_TIMEOUT


def get_base_url():
    """Return an optional base URL override, or None to use the SDK default."""
    return _clean(os.getenv("GROQ_BASE_URL")) or None


def sdk_available() -> bool:
    return Groq is not None


def is_configured() -> bool:
    """True when the SDK is installed AND an API key is present."""
    return sdk_available() and bool(get_api_key())


def _create_client(api_key: str, timeout: float, base_url=None):
    """Create a Groq client. Separated out so tests can patch it."""
    if Groq is None:
        raise LLMUnavailable(
            "The 'groq' package is not installed. Run "
            "'python -m pip install -r requirements.txt' and restart the server.",
        )
    api_key = _clean(api_key)
    if not api_key:
        raise LLMNotConfigured(
            "GROQ_API_KEY is missing or empty. Set it in your local .env "
            "and restart the server.",
        )
    kwargs = {"api_key": api_key, "timeout": timeout}
    if base_url:
        kwargs["base_url"] = base_url
    try:
        return Groq(**kwargs)
    except Exception as exc:  # pragma: no cover - construction rarely fails
        raise LLMUnavailable(
            f"Could not initialize the Groq client ({type(exc).__name__}).",
        ) from exc


def get_client():
    """Return a cached Groq client, creating it on first use."""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            _client = _create_client(get_api_key(), get_timeout(), get_base_url())
    return _client


def reset_client() -> None:
    """Drop the cached client (used by tests and after config changes)."""
    global _client
    with _client_lock:
        _client = None


def provider_status() -> dict:
    """Diagnostic snapshot. Contains NO secret values."""
    return {
        "sdk_installed": sdk_available(),
        "configured": is_configured(),
        "model": get_model(),
        "base_url": get_base_url() or "https://api.groq.com (Groq SDK default)",
        "timeout": get_timeout(),
    }


def _describe_error(exc) -> str:
    """Map a Groq SDK exception to a safe, actionable message."""
    model = get_model()
    if groq_sdk is not None:
        if isinstance(exc, groq_sdk.AuthenticationError):
            return (
                "Groq rejected the API key (HTTP 401). Check GROQ_API_KEY in "
                "your local .env and restart the server."
            )
        if isinstance(exc, groq_sdk.PermissionDeniedError):
            return (
                "Groq denied access for this API key (HTTP 403). Verify the key's "
                "permissions and model access."
            )
        if isinstance(exc, groq_sdk.RateLimitError):
            return (
                "Groq rate limit reached (HTTP 429). Wait a moment and retry, or "
                "reduce the request rate."
            )
        if isinstance(exc, groq_sdk.APITimeoutError):
            return (
                f"The Groq request timed out after {get_timeout():.0f}s. Check "
                "connectivity or increase GROQ_TIMEOUT."
            )
        if isinstance(exc, groq_sdk.APIConnectionError):
            return (
                "Could not reach the Groq API (connection error). Check network "
                "access and retry."
            )
        if isinstance(exc, (groq_sdk.BadRequestError, groq_sdk.NotFoundError)):
            return (
                f"Groq rejected model '{model}' (HTTP "
                f"{getattr(exc, 'status_code', '4xx')}). It may be an invalid or "
                "retired model id; set GROQ_MODEL to a supported model such as "
                f"'{DEFAULT_GROQ_MODEL}'."
            )
        if isinstance(exc, groq_sdk.APIStatusError):
            return (
                f"Groq returned an error (HTTP {getattr(exc, 'status_code', '?')}) "
                "for model "
                f"'{model}'."
            )
    return f"Groq request failed ({type(exc).__name__})."


def chat_completion(messages, *, temperature=0.6, max_tokens=200, model=None):
    """Run one chat completion against Groq and return the reply text.

    Raises LLMNotConfigured when no key is set and LLMUnavailable when the
    call fails, so callers can fall back to a deterministic local reply.
    """
    client = get_client()
    if groq_sdk is not None:
        error_types = (groq_sdk.GroqError,)
    else:  # pragma: no cover - sdk_available() would already be False
        error_types = (Exception,)
    try:
        response = client.chat.completions.create(
            model=model or get_model(),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except error_types as exc:
        raise LLMUnavailable(_describe_error(exc)) from exc
    return response.choices[0].message.content.strip()
