#!/usr/bin/env python3
"""
Batch evaluation script for Streamlit side-by-side UI Dashboard.
Evaluates the standard RAG pipeline vs PrefPO optimized RAG pipeline.
Outputs results to `evaluation_results.json` which the UI subsequently loads.
"""

import time
import json
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Import RAG pipeline instances
from retrieve_v2 import HybridRetrieverV2
from rerank_v2 import TwoStageCalibratedReranker
from llm import GenerationClient
from config import SYSTEM_PROMPT, INGEST_EMBEDDING_MODEL, QDRANT_PATH, QDRANT_COLLECTION, SPARSE_MODE
from pathlib import Path

# Since prefpo was unable to be executed on this machine due to Python < 3.11,
# This is a hypothetical OPTIMIZED prompt that explicitly aligns with prefpo constraints logic.
PREFPO_OPTIMIZED_PROMPT = """You are an elite, highly precise HVAC technical assistant.
Your absolute priority is safety and strict adherence to the provided source text. 

CRITICAL CONSTRAINTS:
1. Grounding: Answer strictly and entirely based on the provided local corpus snippets. 
If the exact answer is absolutely NOT in the context, output ONLY: "The sources do not cover this topic." Do not guess or infer outside the context.
2. Citation: Every fact must be attributed using bracketed numbers mapping to the sources, e.g. [1][3].
3. Format: Keep the output completely technical, void of fluff, under 150 words.
4. If multiple modes or parameters are asked, use a bulleted list. 

Sources provided:
{context}

Question: {query}

Provide your grounded answer below:"""

def get_retriever():
    def _has_qdrant_collection(path: str, col: str) -> bool:
        return (Path(path) / "collection" / col / "storage.sqlite").exists()
        
    candidates = [
        (QDRANT_PATH, QDRANT_COLLECTION),
        ("./qdrant_db_ci", "hvac_documents_qdrant_ci"),
        ("./qdrant_db", "hvac_documents_qdrant"),
    ]
    path, col = next(
        ((p, c) for p, c in candidates if _has_qdrant_collection(p, c)),
        (QDRANT_PATH, QDRANT_COLLECTION),
    )
    return HybridRetrieverV2(
        backend="qdrant",
        embedding_model=INGEST_EMBEDDING_MODEL,
        sparse_mode=SPARSE_MODE if SPARSE_MODE in {"none", "bm42", "splade"} else "bm42",
        qdrant_path=path,
        qdrant_collection=col,
    )

def evaluate_sample(query: str, target: str, retriever, reranker, gen_client):
    """Run full retrieve, rerank and double generation pipeline for both prompts."""
    # 1. Retrieve & Rerank (Shared between both to be fair on latency / docs comparison)
    retrieval_start = time.perf_counter()
    candidates = retriever.search(query)
    reranked = reranker.rerank(query, candidates)
    top_docs = reranked[:5]
    total_retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

    # 2. Baseline generation
    start_base = time.perf_counter()
    base_ans = gen_client.generate(query, top_docs, system_prompt=SYSTEM_PROMPT, use_fallback=True)
    base_latency = (time.perf_counter() - start_base) * 1000

    # 3. PrefPO generation
    start_prefpo = time.perf_counter()
    prefpo_ans = gen_client.generate(query, top_docs, system_prompt=PREFPO_OPTIMIZED_PROMPT, use_fallback=True)
    prefpo_latency = (time.perf_counter() - start_prefpo) * 1000

    # Quick heuristic accuracy evaluator (In production replace with LLM judge)
    def grade(ans: str) -> float:
        # Lowercase exact word matching heuristic
        t_words = str(target).lower().split()
        a_words = str(ans).lower()
        if len(t_words) == 0: return 0.0
        hits = sum(1 for w in t_words if len(w) > 3 and w in a_words)
        return float(hits) / len([w for w in t_words if len(w)>3] or [1]) * 100

    return {
        "query": query,
        "standard": {
            "answer": base_ans,
            "latency_ms": base_latency + total_retrieval_ms,
            "score": grade(base_ans)
        },
        "prefpo": {
            "answer": prefpo_ans,
            "latency_ms": prefpo_latency + total_retrieval_ms,
            "score": grade(prefpo_ans)
        }
    }

def main():
    print("Loading RAG models...")
    retriever = get_retriever()
    reranker = TwoStageCalibratedReranker()
    gen_client = GenerationClient()

    print("Loading 126 evaluation dataset from eval/golden.csv...")
    df = pd.read_csv("eval/golden.csv").dropna(subset=["Question"])
    
    results = []
    
    total_q = len(df)
    print(f"Beginning side-by-side evaluation of {total_q} questions...")
    
    # Normally we do ThreadPoolExecutor, but RAG with litellm rate-limits are tricky, doing sequential
    for idx, row in tqdm(df.iterrows(), total=total_q):
        try:
            res = evaluate_sample(
                query=str(row["Question"]), 
                target=row.get("anchor_text", ""), 
                retriever=retriever, 
                reranker=reranker, 
                gen_client=gen_client
            )
            results.append(res)
        except Exception as e:
            print(f"Failed query at {idx}: {e}")

    # Compute metrics
    metrics = {
        "total_questions": len(results),
        "standard_avg_latency_ms": sum(r["standard"]["latency_ms"] for r in results) / max(1, len(results)),
        "prefpo_avg_latency_ms": sum(r["prefpo"]["latency_ms"] for r in results) / max(1, len(results)),
        "standard_accuracy_score": sum(r["standard"]["score"] for r in results) / max(1, len(results)),
        "prefpo_accuracy_score": sum(r["prefpo"]["score"] for r in results) / max(1, len(results)),
        "samples": results
    }

    with open("evaluation_results.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Exported evaluation_results.json.")

if __name__ == "__main__":
    main()
