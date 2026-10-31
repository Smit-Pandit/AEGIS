import os
import sys
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Available and verified Groq models
GROQ_MODELS = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]

# Available and verified Gemini models with active free-tier quota
GEMINI_MODELS = [
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-3.5-flash-lite",
    "gemini-3.7-flash",
    "gemma-4-26b-a4b-it",
]


def call_groq_chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 1500,
    timeout: int = 25,
) -> str:
    """Calls Groq API with model failovers."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not configured in .env")

    models_to_try = [model] if model else GROQ_MODELS
    last_err = None

    for m in models_to_try:
        try:
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                },
                data=json.dumps({
                    "model": m,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }).encode("utf-8"),
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            last_err = f"Groq HTTP {e.code} for model {m}: {err_body}"
            if e.code == 429:
                import time
                time.sleep(1.5)
            continue
        except Exception as e:
            last_err = f"Groq Exception for model {m}: {str(e)}"
            continue

    raise RuntimeError(f"All Groq models failed. Last error: {last_err}")


def call_gemini_generate(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.1,
    timeout: int = 25,
) -> str:
    """Calls Gemini GenerateContent API with model failovers."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured in .env")

    models_to_try = [model] if model else GEMINI_MODELS
    last_err = None

    for m in models_to_try:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={GEMINI_API_KEY}"
            req = urllib.request.Request(
                url,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                },
                data=json.dumps({
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": temperature},
                }).encode("utf-8"),
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            last_err = f"Gemini HTTP {e.code} for model {m}: {err_body}"
            continue
        except Exception as e:
            last_err = f"Gemini Exception for model {m}: {str(e)}"
            continue

    raise RuntimeError(f"All Gemini models failed. Last error: {last_err}")


def call_gemini_multimodal(
    image_base64: str,
    mime_type: str = "image/png",
    prompt: str = "Describe this image in detail, identifying any entity names, species, metrics, text, and visual attributes.",
    timeout: int = 25,
) -> str:
    """Calls Gemini with image payload to extract semantic descriptions for multimodal indexing."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is required for multimodal image comprehension.")

    for m in GEMINI_MODELS:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={GEMINI_API_KEY}"
            req = urllib.request.Request(
                url,
                headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
                data=json.dumps({
                    "contents": [{
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": mime_type, "data": image_base64}},
                        ]
                    }],
                    "generationConfig": {"temperature": 0.1},
                }).encode("utf-8"),
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            print(f"[Gemini Multimodal] Model {m} error: {e}")
            continue

    return f"Image asset (format: {mime_type})"


def llm_generate(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 1500,
) -> str:
    """Unified LLM generator with Groq as primary and Gemini as secondary."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # 1. Try Groq first
    if GROQ_API_KEY:
        try:
            res = call_groq_chat(messages, temperature=temperature, max_tokens=max_tokens)
            if res and res.strip():
                return res
        except Exception as e:
            print(f"[LLM] Groq attempt failed ({e}), attempting Gemini fallback...")

    # 2. Try Gemini fallback
    if GEMINI_API_KEY:
        try:
            full_prompt = f"{system_prompt}\n\nUser Request:\n{user_prompt}"
            res = call_gemini_generate(full_prompt, temperature=temperature)
            if res and res.strip():
                return res
        except Exception as e:
            print(f"[LLM] Gemini attempt failed ({e}).")

    raise RuntimeError("Both Groq and Gemini API calls failed. Please check API keys and connectivity.")


def llm_json(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 1500,
) -> Dict[str, Any]:
    """Generates and safely parses JSON from the LLM."""
    json_system = f"{system_prompt}\n\nIMPORTANT: You must reply ONLY with a valid JSON object. Do not include markdown code block syntax (like ```json), commentary, or any text before or after the JSON."
    text = llm_generate(json_system, user_prompt, temperature=temperature, max_tokens=max_tokens)

    # Clean markdown fences if present
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Clean any bold/italic/bullet markdown markers before or after JSON keys/values
    import re
    cleaned = re.sub(r'[*_]+(?=")', '', cleaned)
    cleaned = re.sub(r'(?<=")[*_]+', '', cleaned)
    cleaned = re.sub(r'^\s*[*+-]\s+(?=")', '', cleaned, flags=re.MULTILINE)
    cleaned = cleaned.replace("**", "")

    # Locate first { and last }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start : end + 1]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as err:
        print(f"[LLM JSON Error] Raw text was:\n{text}")
        raise RuntimeError(f"Failed to parse LLM response as JSON: {err}")
