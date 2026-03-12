"""Generation client using Groq with local Ollama fallback."""

import os
import requests
from groq import Groq
from config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TOP_P, SYSTEM_PROMPT, SYSTEM_MESSAGE

OLLAMA_URL = "http://localhost:11434/api/generate"

class GenerationClient:
    """Handles LLM interactions for both Groq and local Ollama fallbacks."""

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.groq = Groq(api_key=self.api_key) if self.api_key else None

    def _build_context(self, hits: list) -> str:
        """Format retrieval hits into a structured context string."""
        parts = []
        for i, hit in enumerate(hits, 1):
            meta = hit["metadata"]
            parts.append(
                f"[Source {i}] (Document: {meta['filename']}, Page: {meta['page_number']})\n"
                f"{hit['document']}\n"
            )
        return "\n---\n".join(parts)

    def generate(self, query: str, hits: list, use_fallback: bool = True) -> str:
        """Generate a safety-compliant answer from provided sources."""
        context = self._build_context(hits)
        prompt = SYSTEM_PROMPT.format(context=context, query=query)

        if self.groq:
            try:
                res = self.groq.chat.completions.create(
                    messages=[
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user", "content": prompt},
                    ],
                    model=LLM_MODEL,
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                    top_p=LLM_TOP_P,
                )
                return res.choices[0].message.content
            except Exception as e:
                print(f"Groq API error: {e}")
                if not use_fallback: return f"Error: {e}"

        return self._generate_ollama(prompt)

    def _generate_ollama(self, prompt: str) -> str:
        """Fallback to local Ollama if Groq fails or is unavailable."""
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
        try:
            res = requests.post(
                OLLAMA_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "system": SYSTEM_MESSAGE,
                    "stream": False,
                    "options": {"temperature": LLM_TEMPERATURE}
                },
                timeout=60
            )
            res.raise_for_status()
            return res.json().get("response", "").strip()
        except Exception as e:
            return f"Service unavailable: {e}"
