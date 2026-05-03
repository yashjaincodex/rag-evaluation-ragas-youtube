import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas import SingleTurnSample
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,  # Did the generated response is related to the question or not - Detects off-topic answers
)
from ragas.metrics.collections import (
    ContextRecall,  # Did the retrived context is correct and relevant to query - Detects retrieval failure
)
from ragas.metrics.collections import (
    FactualCorrectness,  # Did the generated response is correct or not - Detects wrong answers
)
from ragas.metrics.collections import (
    Faithfulness,  # Did LLM generated response using the retrieved info or created on its own - Detects hallucination
)

load_dotenv()

# ── LLM + Embeddings ──────────────────────────────────────────
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)
embeddings = OpenAIEmbeddings(client=client, model="text-embedding-3-small")

# ─────────────────────────────────────────────────────────────
# SAMPLE 1 — BAD Faithfulness (Hallucination)
# The LLM made up a fact NOT present in the context.
# Context only says "Paris is the capital of France."
# But the response adds "Paris has a population of 5 million" — hallucinated!
# Expected: Faithfulness LOW
# ─────────────────────────────────────────────────────────────
sample_hallucination = SingleTurnSample(
    user_input="What is the capital of France?",
    retrieved_contexts=["Paris is the capital of France."],
    response="Paris is the capital of France. Paris has a population of 5 million people.",
    reference="Paris is the capital of France.",
)

# ─────────────────────────────────────────────────────────────
# SAMPLE 2 — BAD Context Recall (Retriever missed key info)
# The reference answer mentions Einstein won the Nobel Prize in 1921.
# But the retrieved context only talks about his theory of relativity —
# it does NOT mention the Nobel Prize at all.
# Expected: Context Recall LOW
# ─────────────────────────────────────────────────────────────
sample_bad_retrieval = SingleTurnSample(
    user_input="What is Albert Einstein known for?",
    retrieved_contexts=["Albert Einstein developed the theory of relativity."],
    response="Einstein is known for the theory of relativity.",
    reference="Einstein is known for the theory of relativity and winning the Nobel Prize in Physics in 1921.",
)

# ─────────────────────────────────────────────────────────────
# SAMPLE 3 — BAD Factual Correctness (Wrong answer)
# The LLM confidently gave the wrong year.
# Bell invented the telephone in 1876, not 1901.
# Expected: Factual Correctness LOW
# ─────────────────────────────────────────────────────────────
sample_wrong_answer = SingleTurnSample(
    user_input="When did Alexander Graham Bell invent the telephone?",
    retrieved_contexts=["Alexander Graham Bell invented the telephone in 1876."],
    response="Alexander Graham Bell invented the telephone in 1901.",
    reference="Alexander Graham Bell invented the telephone in 1876.",
)

# ─────────────────────────────────────────────────────────────
# SAMPLE 4 — BAD Answer Relevancy (Off-topic answer)
# User asked about Python the programming language.
# The LLM answered about Python the snake — completely off-topic!
# Expected: Answer Relevancy LOW
# ─────────────────────────────────────────────────────────────
sample_offtopic = SingleTurnSample(
    user_input="What is Python used for in software development?",
    retrieved_contexts=["Python is a popular programming language used for web development, data science, and AI."],
    response="Python is a large non-venomous snake found in Africa and Asia. It is known for squeezing its prey.",
    reference="Python is used for web development, data science, machine learning, and automation.",
)

# ─────────────────────────────────────────────────────────────
# SAMPLE 5 — GOOD (All metrics high — baseline for comparison)
# ─────────────────────────────────────────────────────────────
sample_good = SingleTurnSample(
    user_input="What is machine learning?",
    retrieved_contexts=[
        "Machine learning is a subset of AI that enables systems to learn from data and improve over time."
    ],
    response="Machine learning is a subset of AI that allows systems to learn from data and improve over time.",
    reference="Machine learning is a subset of AI that enables systems to learn from data and improve over time.",
)

# ── Metrics ───────────────────────────────────────────────────
faithfulness = Faithfulness(llm=llm)
context_recall = ContextRecall(llm=llm)
factual_correctness = FactualCorrectness(llm=llm)
answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)

SAMPLES = [
    ("🔴 Hallucination (bad Faithfulness)", sample_hallucination),
    ("🔴 Retrieval Miss (bad Context Recall)", sample_bad_retrieval),
    ("🔴 Wrong Answer (bad Factual Correctness)", sample_wrong_answer),
    ("🔴 Off-Topic Answer (bad Answer Relevancy)", sample_offtopic),
    ("🟢 Good Answer (all metrics high)", sample_good),
]


async def evaluate_all():
    print("\n" + "=" * 55)
    print("  RAGAS — Hallucination & Error Detection")
    print("=" * 55)

    for label, sample in SAMPLES:
        print(f"\n{label}")
        print(f"  Q: {sample.user_input}")
        print(f"  A: {sample.response}")
        print(f"  {'─' * 50}")

        f_result = await faithfulness.ascore(
            user_input=sample.user_input,
            response=sample.response,
            retrieved_contexts=sample.retrieved_contexts,
        )
        cr_result = await context_recall.ascore(
            user_input=sample.user_input,
            retrieved_contexts=sample.retrieved_contexts,
            reference=sample.reference,
        )
        fc_result = await factual_correctness.ascore(
            response=sample.response,
            reference=sample.reference,
        )
        ar_result = await answer_relevancy.ascore(
            user_input=sample.user_input,
            response=sample.response,
        )

        scores = {
            "Faithfulness       (hallucination?)": f_result.value,
            "Context Recall     (retrieval ok?) ": cr_result.value,
            "Factual Correctness(answer correct?)": fc_result.value,
            "Answer Relevancy   (on-topic?)     ": ar_result.value,
        }

        for metric, value in scores.items():
            emoji = "✅" if value >= 0.8 else "⚠️" if value >= 0.5 else "❌"
            print(f"  {emoji}  {metric}  {value:.4f}")


asyncio.run(evaluate_all())
