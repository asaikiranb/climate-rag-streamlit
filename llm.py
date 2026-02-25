"""Groq client and answer generation (with Ollama fallback)."""

import os
import subprocess
import time
import requests
from dotenv import load_dotenv
from groq import Groq

from config import SYSTEM_PROMPT, SYSTEM_MESSAGE, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TOP_P

load_dotenv()

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

_ollama_started = False


def _ensure_ollama_running():
    """Start Ollama server if not already running."""
    global _ollama_started
    if _ollama_started:
        return
    try:
        requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        _ollama_started = True
        return
    except Exception:
        pass
    print("    Starting Ollama server for answer generation...")
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(30):
        time.sleep(1)
        try:
            requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
            _ollama_started = True
            print("    Ollama server ready")
            return
        except Exception:
            pass

def get_groq_client() -> Groq:
    """Return a Groq client. Raises ValueError if API key is missing."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Please add it to your .env file.")
    return Groq(api_key=api_key)


def build_context(results: list) -> str:
    """Turn search results into a numbered source string for the prompt."""
    context_parts = []
    for i, result in enumerate(results, 1):
        metadata = result['metadata']
        context_parts.append(
            f"[Source {i}] (Document: {metadata['filename']}, Page: {metadata['page_number']})\n"
            f"{result['document']}\n"
        )
    return "\n---\n".join(context_parts)


def generate_answer(query: str, context: str, groq_client: Groq) -> str:
    """Send query + context to Groq and return the answer."""
    prompt = SYSTEM_PROMPT.format(context=context, query=query)

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            top_p=LLM_TOP_P,
        )
        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"Error generating answer: {str(e)}"


def generate_answer_ollama(query: str, context: str) -> str:
    """Send query + context to local Ollama and return the answer."""
    _ensure_ollama_running()
    prompt = SYSTEM_PROMPT.format(context=context, query=query)

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "system": SYSTEM_MESSAGE,
                "stream": False,
                "options": {
                    "temperature": LLM_TEMPERATURE,
                    "num_predict": LLM_MAX_TOKENS,
                    "top_p": LLM_TOP_P,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    except Exception as e:
        return f"Error generating answer: {str(e)}"
