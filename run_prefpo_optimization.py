#!/usr/bin/env python3
"""
Optimization script for RAG-climate using PrefPO.
This script demonstrates how to leverage the PrefPO library in instruction mode 
to optimize the `SYSTEM_PROMPT` used for generating answers in RAG-climate.
"""

import os
import asyncio
import pandas as pd
from typing import List

try:
    from prefpo import PrefPOConfig, Grader, GradeResult, optimize_async
    from prefpo.generate import generate_outputs
    from prefpo.types import Sample
    from prefpo import call_llm
except ImportError:
    print("Warning: prefpo is not installed or using incompatible python. Needs python >= 3.11. Simulation mode active.")
    PrefPOConfig = None

from config import SYSTEM_PROMPT

# Prepare dataset
def load_dataset() -> List:
    df = pd.read_csv("eval/golden.csv")
    samples = []
    for idx, row in df.iterrows():
        samples.append(
            # Since prefpo isn't installed locally we mock sample struct if it fails
            {"index": idx, "question": row["Question"], "target": row["anchor_text"]}
        )
    return samples

class RAGLLMJudgeGrader:
    """A custom Grader for PrefPO that evaluates how well the generated answer adheres to the gold anchor_text."""
    async def grade(self, prompt, samples, model_config, semaphore):
        outputs = await generate_outputs(prompt, samples, model_config, semaphore)
        score = 0
        for o, s in zip(outputs, samples):
            judge_resp = await call_llm(
                model="groq/llama-3.3-70b-versatile",
                messages=[{
                    "role": "user", 
                    "content": f"Evaluate this RAG answer.\n\nQuestion: {s.question}\nTarget Fact: {s.target}\nAnswer: {o.response}\n\nDoes the answer accurately reflect the Target Fact without adding hallucinations? Respond strictly with '1' if Yes, or '0' if No. Score 0 or 1:"
                }],
            )
            score += int("1" in judge_resp.output_text)
        return GradeResult(score=score / len(outputs), n=len(outputs))

async def main():
    if PrefPOConfig is None:
        print("Cannot run PrefPO optimization locally on this environment without Python >= 3.11. Please upgrade and pip install -e ../prefpo.")
        return

    samples = load_dataset()
    train = samples[:20] # Take a subset to be fast

    config = PrefPOConfig(
        mode="instruction",
        task_model={"name": "groq/llama-3.3-70b-versatile"},
        discriminator={
            "criteria": [
                "Must be highly factual, answering the specific question.",
                "Should securely cite sources in an array like [1].",
                "Must be concise, under 150 words.",
                "Failure to follow format or inserting hallucinations is severely penalized."
            ]
        },
        optimizer={
            "model": {"name": "groq/llama-3.3-70b-versatile"}
        },
        pool={"initial_prompts": [SYSTEM_PROMPT]},
        run={"iterations": 5, "output_dir": "results/prefpo_rag"},
    )

    print("Running optimization using groq/llama-3.3-70b-versatile...")
    result = await optimize_async(config, grader=RAGLLMJudgeGrader(), train=train)
    
    print(f"Best score: {result.best_score:.3f}")
    print(f"Best prompt:\n{result.best_prompt.value}")

    with open("optimized_prompt.txt", "w") as f:
        f.write(result.best_prompt.value)

if __name__ == "__main__":
    asyncio.run(main())
