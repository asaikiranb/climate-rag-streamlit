"""Generation client using Groq with local Ollama fallback."""

import os
import re
import requests
from groq import Groq
from config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TOP_P,
    SYSTEM_PROMPT,
    SYSTEM_MESSAGE,
    WEB_FALLBACK_ENABLED,
    WEB_FALLBACK_MAX_SNIPPETS,
)

OLLAMA_URL = "http://localhost:11434/api/generate"
TAVILY_SEARCH_API_URL = "https://api.tavily.com/search"

WEB_REWRITE_PROMPT = """You are a grounded HVAC assistant.
Use BOTH sources below:
1) Retrieved local corpus context
2) Supplemental web snippets (for freshness)

Rules:
- Prefer local corpus for domain/safety guidance.
- Use web snippets only for time-sensitive/current facts.
- If web snippets conflict with local context, call that out clearly.
- Cite local sources as [1], [2], etc. and web snippets as [W1], [W2], etc.

Local context:
{context}

Web snippets:
{web_context}

Question: {query}

Answer:"""

QUERY_RECONSTRUCT_PROMPT = """You improve retrieval queries for HVAC document search.

Input question:
{question}

Manual query draft:
{manual_query}

OCR text (optional):
{ocr_text}

Structured fields (optional):
{fields_text}

Detected objects (optional):
{objects_text}

Task:
1) Produce ONE concise, deduplicated search query in English.
2) Preserve technical tokens exactly (model numbers, refrigerants, voltages, standards).
3) Avoid filler words and duplicates.
4) Keep it under 45 words.

Return only the rewritten query text, no markdown or bullets.
"""

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

    def _chat_completion(self, prompt: str, use_fallback: bool = True) -> str:
        """Run generation via Groq first, then Ollama fallback if enabled."""
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
                if not use_fallback:
                    return f"Error: {e}"

        return self._generate_ollama(prompt)

    def _heuristic_reconstruct_query(
        self,
        question: str,
        manual_query: str,
        ocr_text: str = "",
        fields: dict | None = None,
        objects: list[dict] | None = None,
    ) -> str:
        """Deterministic fallback query cleanup when LLM rewrite is unavailable."""
        fields = fields or {}
        objects = objects or []

        components = [manual_query.strip() or question.strip()]
        if fields:
            field_tokens = [f"{key} {value}" for key, value in fields.items() if value]
            if field_tokens:
                components.append(" ".join(field_tokens))
        if objects:
            labels = [str(item.get("label", "")).strip() for item in objects if item.get("label")]
            if labels:
                components.append("objects " + " ".join(labels))
        if ocr_text.strip():
            components.append(ocr_text.strip()[:600])

        raw = " ".join(part for part in components if part)
        tokens = re.findall(r"[A-Za-z0-9\-./]+", raw)
        deduped: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            key = token.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(token)
            if len(deduped) >= 55:
                break
        return " ".join(deduped)

    def reconstruct_query_with_metadata(
        self,
        question: str,
        manual_query: str,
        ocr_text: str = "",
        fields: dict | None = None,
        objects: list[dict] | None = None,
    ) -> dict:
        """Reconstruct query using one LLM call with deterministic fallback."""
        fields = fields or {}
        objects = objects or []
        fallback_query = self._heuristic_reconstruct_query(
            question=question,
            manual_query=manual_query,
            ocr_text=ocr_text,
            fields=fields,
            objects=objects,
        )

        if not self.groq:
            return {
                "reconstructed_query": fallback_query,
                "used_llm": False,
                "reason": "groq_unavailable",
                "fallback_query": fallback_query,
            }

        try:
            prompt = QUERY_RECONSTRUCT_PROMPT.format(
                question=question or "",
                manual_query=manual_query or "",
                ocr_text=(ocr_text or "")[:1800],
                fields_text=str(fields)[:900],
                objects_text=str(objects)[:700],
            )
            result = self.groq.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You optimize retrieval queries for technical RAG systems."},
                    {"role": "user", "content": prompt},
                ],
                model=LLM_MODEL,
                temperature=0.1,
                max_tokens=120,
                top_p=0.9,
            )
            candidate = (result.choices[0].message.content or "").strip()
            if not candidate:
                raise ValueError("empty query rewrite")
            first_line = candidate.splitlines()[0].strip()
            reconstructed = first_line.strip('"').strip()
            if not reconstructed:
                raise ValueError("blank query rewrite")
            return {
                "reconstructed_query": reconstructed,
                "used_llm": True,
                "reason": "ok",
                "fallback_query": fallback_query,
            }
        except Exception as exc:
            return {
                "reconstructed_query": fallback_query,
                "used_llm": False,
                "reason": f"fallback:{exc}",
                "fallback_query": fallback_query,
            }

    def _should_use_web_fallback(self, query: str, answer: str) -> bool:
        """Heuristic web-fallback trigger for fresh/news-like or weak-answer queries."""
        query_text = (query or "").lower()
        answer_text = (answer or "").lower()

        recency_terms = {
            "latest",
            "today",
            "recent",
            "new",
            "news",
            "current",
            "update",
            "announcement",
            "this year",
            "2025",
            "2026",
        }
        uncertainty_markers = {
            "do not have",
            "not available",
            "not covered",
            "insufficient",
            "service unavailable",
            "could not",
        }

        has_recency_signal = any(term in query_text for term in recency_terms)
        has_uncertainty = any(term in answer_text for term in uncertainty_markers)
        return has_recency_signal or has_uncertainty

    def _fetch_web_snippets(self, query: str, max_snippets: int = WEB_FALLBACK_MAX_SNIPPETS) -> list[dict]:
        """Fetch compact web snippets via Tavily search API."""
        provider = (os.getenv("WEB_SEARCH_PROVIDER") or "tavily").strip().lower()
        if provider and provider != "tavily":
            return []

        api_key = os.getenv("TAVILY_API_KEY") or os.getenv("WEB_SEARCH_API_KEY")
        if not api_key:
            return []

        snippets: list[dict] = []
        try:
            response = requests.post(
                TAVILY_SEARCH_API_URL,
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": max_snippets,
                    "include_answer": False,
                    "include_raw_content": False,
                },
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return snippets

        results = payload.get("results") or []
        for item in results:
            if len(snippets) >= max_snippets:
                break
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            snippets.append(
                {
                    "title": title or "Tavily result",
                    "url": url,
                    "snippet": content,
                }
            )

        deduped: list[dict] = []
        seen = set()
        for snippet in snippets:
            key = re.sub(r"\s+", " ", (snippet.get("snippet") or "").strip().lower())
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(snippet)
            if len(deduped) >= max_snippets:
                break
        return deduped

    def generate_with_metadata(
        self,
        query: str,
        hits: list,
        use_fallback: bool = True,
        extra_context: str = "",
        allow_web_fallback: bool = True,
    ) -> dict:
        """Generate answer with optional web augmentation and metadata for observability."""
        context = self._build_context(hits)
        if extra_context.strip():
            context = f"{context}\n\n[Input Signals]\n{extra_context.strip()}"

        prompt = SYSTEM_PROMPT.format(context=context, query=query)
        answer = self._chat_completion(prompt, use_fallback=use_fallback)

        web_snippets: list[dict] = []
        web_used = False
        if WEB_FALLBACK_ENABLED and allow_web_fallback and self._should_use_web_fallback(query, answer):
            web_snippets = self._fetch_web_snippets(query)
            if web_snippets:
                web_context = "\n".join(
                    f"[W{i}] {snippet['title']}\nURL: {snippet['url']}\n{snippet['snippet']}"
                    for i, snippet in enumerate(web_snippets, 1)
                )
                web_prompt = WEB_REWRITE_PROMPT.format(
                    context=context,
                    web_context=web_context,
                    query=query,
                )
                answer = self._chat_completion(web_prompt, use_fallback=use_fallback)
                web_used = True

        return {
            "answer": answer,
            "web_used": web_used,
            "web_snippets": web_snippets,
        }

    def generate(self, query: str, hits: list, use_fallback: bool = True) -> str:
        """Generate a safety-compliant answer from provided sources."""
        result = self.generate_with_metadata(query, hits, use_fallback=use_fallback)
        return result["answer"]

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
