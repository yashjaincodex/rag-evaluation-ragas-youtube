import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

load_dotenv()

# ── LLM + Embeddings ──────────────────────────────────────────
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)
embeddings = OpenAIEmbeddings(client=client, model="text-embedding-3-small")

# ── Metrics ───────────────────────────────────────────────────
faithfulness = Faithfulness(llm=llm)
answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)

# ─────────────────────────────────────────────────────────────
# DUMMY RAG
# Simulates a retriever by doing a naive keyword match against
# a small in-memory knowledge base.
# In production, replace retrieve() with your real vector DB call.
# ─────────────────────────────────────────────────────────────
KNOWLEDGE_BASE = [
    "Paris is the capital of France and has a population of about 2.1 million in the city proper.",
    "Albert Einstein developed the theory of relativity and won the Nobel Prize in Physics in 1921.",
    "Alexander Graham Bell invented the telephone in 1876.",
    "Python is a popular programming language used for web development, data science, and AI.",
    "Machine learning is a subset of AI that enables systems to learn from data and improve over time.",
    "The Eiffel Tower is located in Paris and was constructed in 1889.",
    "The speed of light in a vacuum is approximately 299,792 km/s.",
    "Photosynthesis is the process by which plants convert sunlight into food using CO2 and water.",
    "The Great Wall of China stretches over 21,000 kilometers.",
    "Shakespeare wrote 37 plays and 154 sonnets during his lifetime.",
]


def retrieve(query: str, top_k: int = 2) -> list[str]:
    """
    Dummy retriever: scores each document by counting how many
    query words appear in it, then returns the top_k matches.
    Replace this function with your actual vector search.
    """
    query_words = set(query.lower().split())
    scored = [(doc, sum(1 for w in query_words if w in doc.lower())) for doc in KNOWLEDGE_BASE]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in scored[:top_k] if score > 0]
    # Fallback: return top_k docs even if no keyword match
    if not top_docs:
        top_docs = [doc for doc, _ in scored[:top_k]]
    return top_docs


# ─────────────────────────────────────────────────────────────
# DUMMY LLM RESPONSE GENERATOR
# Simulates your RAG's LLM answer step.
# In production, replace generate_response() with your actual
# LLM call that uses the retrieved context to answer the query.
# ─────────────────────────────────────────────────────────────
async def generate_response(query: str, contexts: list[str]) -> str:
    """
    Generates an answer using GPT given the query and retrieved contexts.
    This mirrors what your production RAG pipeline would do.
    """
    context_text = "\n".join(f"- {c}" for c in contexts)
    prompt = (
        f"Answer the following question using ONLY the provided context.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        f"Answer concisely in 1-2 sentences."
    )
    result = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    return result.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────
# SCORER
# ─────────────────────────────────────────────────────────────
async def score_query(query: str) -> None:
    print(f"\n{'=' * 58}")
    print(f"  Query: {query}")
    print(f"{'=' * 58}")

    # Step 1 — Retrieve
    contexts = retrieve(query)
    print("\n📄 Retrieved Context(s):")
    for i, ctx in enumerate(contexts, 1):
        print(f"   {i}. {ctx}")

    # Step 2 — Generate response
    response = await generate_response(query, contexts)
    print(f"\n🤖 Generated Response:\n   {response}")

    # Step 3 — Score
    print("\n📊 RAGAS Scores:")
    f_result = await faithfulness.ascore(
        user_input=query,
        response=response,
        retrieved_contexts=contexts,
    )
    ar_result = await answer_relevancy.ascore(
        user_input=query,
        response=response,
    )

    score_sum = 0
    for label, value in [
        ("Faithfulness    (hallucination?)", f_result.value),
        ("Answer Relevancy(on-topic?)     ", ar_result.value),
    ]:
        emoji = "✅" if value >= 0.8 else "⚠️" if value >= 0.5 else "❌"
        print(f"   {emoji}  {label}  {value:.4f}")
        score_sum += value

    average_score = score_sum / 2
    emoji = "✅" if average_score >= 0.8 else "⚠️" if average_score >= 0.5 else "❌"
    print(f"   {emoji}  Average Score                     {average_score:.4f}")


# ─────────────────────────────────────────────────────────────
# INTERACTIVE LOOP
# ─────────────────────────────────────────────────────────────
async def main():
    print("\n🔍 RAGAS Live Production Scorer")
    print("   (Faithfulness + Answer Relevancy — no ground truth needed)")
    print("   Type 'quit' or 'exit' to stop.\n")

    while True:
        print("#" * 70)
        query = input("Enter your query: ").strip()
        if not query:
            print("  ⚠️  Please enter a non-empty query.")
            continue
        if query.lower() in {"quit", "exit"}:
            print("\nGoodbye! 👋")
            break
        await score_query(query)


asyncio.run(main())
