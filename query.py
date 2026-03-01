"""Query expansion module used to improve retrieval recall via semantic variations."""

from typing import List
from groq import Groq

from config import LLM_MODEL

EXPANSION_PROMPT = """Generate 2 alternative phrasings of the following question.
The alternatives should use different vocabulary but ask the same thing.
Return only the 2 alternatives, one per line, no numbering, no explanation.

Question: {query}"""


def expand_query(query: str, groq_client: Groq) -> List[str]:
    """Generates alternative phrasings of the input query to broaden retrieval scope.
    
    Returns a list containing the original query and up to two alternatives.
    """
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": EXPANSION_PROMPT.format(query=query)}],
            model=LLM_MODEL,
            temperature=0.5,
            max_tokens=150,
        )
        raw = response.choices[0].message.content.strip()
        alternatives = [line.strip() for line in raw.splitlines() if line.strip()][:2]
        return [query] + alternatives

    except Exception:
        return [query]
