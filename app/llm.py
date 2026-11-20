"""LLM Integration module for Recipe Step Autocompleter.

Provides an optional LLM interface supporting:
- Google Gemini (Gemini API / AI Studio)
- Groq (Ultra-fast Llama-3 inference)
- Ollama (Local laptop model execution)
- Mock / Fallback provider (when offline or no API key is set)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    import requests
except ImportError:
    requests = None  # type: ignore


DEFAULT_RECIPE_SYSTEM_PROMPT = (
    "You are a helpful culinary AI assistant. The user provides a partial cooking recipe instruction. "
    "Autocomplete the step naturally and concisely so the home cook can follow it easily. "
    "Return only the completed recipe step in 1-2 clear, actionable sentences."
)


def get_llm_config() -> Dict[str, str]:
    """Retrieve LLM provider and credentials from environment variables."""
    provider = os.getenv("LLM_PROVIDER", "none").strip().lower()
    return {
        "provider": provider,
        "gemini_api_key": os.getenv("GEMINI_API_KEY", "").strip(),
        "groq_api_key": os.getenv("GROQ_API_KEY", "").strip(),
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip(),
        "model_name": os.getenv("MODEL_NAME", "").strip(),
    }


def is_llm_available() -> bool:
    """Check if an active LLM provider is configured and available."""
    cfg = get_llm_config()
    provider = cfg["provider"]
    if provider == "gemini":
        return bool(cfg["gemini_api_key"])
    elif provider == "groq":
        return bool(cfg["groq_api_key"])
    elif provider == "ollama":
        return True
    elif provider == "mock":
        return True
    return False


def _call_gemini(prompt: str, api_key: str, model: str) -> str:
    """Call Google Gemini API."""
    if not requests:
        raise RuntimeError("The requests library is required for LLM calls.")

    model_name = model or "gemini-2.0-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"{DEFAULT_RECIPE_SYSTEM_PROMPT}\n\nIncomplete recipe step: {prompt}"}
                ]
            }
        ],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 100},
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    candidates = data.get("candidates", [])
    if candidates and "content" in candidates[0]:
        parts = candidates[0]["content"].get("parts", [])
        if parts:
            return parts[0].get("text", "").strip()
    return ""


def _call_groq(prompt: str, api_key: str, model: str) -> str:
    """Call Groq Cloud API."""
    if not requests:
        raise RuntimeError("The requests library is required for LLM calls.")

    model_name = model or "llama-3.1-8b-instant"
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": DEFAULT_RECIPE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Complete this recipe step: {prompt}"},
        ],
        "temperature": 0.5,
        "max_tokens": 100,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _call_ollama(prompt: str, base_url: str, model: str) -> str:
    """Call local Ollama instance."""
    if not requests:
        raise RuntimeError("The requests library is required for LLM calls.")

    model_name = model or "llama3.2"
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model_name,
        "system": DEFAULT_RECIPE_SYSTEM_PROMPT,
        "prompt": f"Complete this recipe step: {prompt}",
        "stream": False,
    }

    resp = requests.post(url, json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


def _call_mock(prompt: str) -> str:
    """Intelligent rule-based fallback completion for offline/testing use."""
    clean = prompt.strip().rstrip(".")
    lower = clean.lower()
    if "heat" in lower and "oil" in lower:
        return f"{clean} until shimmering, about 1 to 2 minutes over medium-high heat."
    elif "preheat" in lower:
        return f"{clean} and line a baking sheet with parchment paper."
    elif "whisk" in lower:
        return f"{clean} in a large mixing bowl until thoroughly combined."
    elif "season" in lower:
        return f"{clean} with kosher salt, freshly ground black pepper, and garlic powder."
    elif "bake" in lower:
        return f"{clean} for 25 to 30 minutes until golden brown and cooked through."
    else:
        return f"{clean} until cooked to desired doneness, stirring occasionally."


def generate_recipe_step_llm(prompt: str) -> str:
    """Complete a recipe step using the configured LLM provider.

    Falls back to mock completion if no provider is configured or on error.
    """
    cfg = get_llm_config()
    provider = cfg["provider"]

    if provider == "gemini" and cfg["gemini_api_key"]:
        try:
            return _call_gemini(prompt, cfg["gemini_api_key"], cfg["model_name"])
        except Exception:
            return _call_mock(prompt)

    elif provider == "groq" and cfg["groq_api_key"]:
        try:
            return _call_groq(prompt, cfg["groq_api_key"], cfg["model_name"])
        except Exception:
            return _call_mock(prompt)

    elif provider == "ollama":
        try:
            return _call_ollama(prompt, cfg["ollama_base_url"], cfg["model_name"])
        except Exception:
            return _call_mock(prompt)

    return _call_mock(prompt)


def llm_hello() -> Dict[str, Any]:
    """Return greeting information and LLM configuration status."""
    cfg = get_llm_config()
    return {
        "greeting": "Hello! Recipe Step Autocompleter is ready to assist home cooks.",
        "provider": cfg["provider"],
        "model": cfg["model_name"] or "Default / Statistical N-Gram",
        "llm_ready": is_llm_available(),
    }
