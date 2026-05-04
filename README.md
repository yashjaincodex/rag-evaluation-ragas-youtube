# 🔍 RAG Evaluation with RAGAS

A simple Python demo to understand how **RAGAS v0.4** evaluates RAG pipelines — detecting hallucinations, wrong answers, retrieval failures, and off-topic responses.

> 📺 **[YouTube Tutorial](https://www.youtube.com/watch?v=ahDqIQb3_8w&t=9s)** · 🐙 **[GitHub Repo](https://github.com/yashjaincodex/rag-evaluation-ragas-youtube)**

---

## 📁 Project Structure

```
rag-evaluation-ragas-youtube/
├── rag_evaluation.py   # Main demo file — all evaluation logic in one place
├── requirements.txt    # Python dependencies
├── .env.example        # API key template
└── README.md

```

---

## 🐍 Install Python Using Miniconda / Miniforge

To keep your AI projects clean and organized, it is recommended to use **conda environments**.

### 🔗 Download Miniforge for macOS (ARM64)

```
https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh

```

> For other platforms (Windows / Linux / Intel Mac), visit the [Miniforge releases page](https://github.com/conda-forge/miniforge/releases).

### 💻 Install Miniforge

```bash
chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh
sh ~/Downloads/Miniforge3-MacOSX-arm64.sh
source ~/miniforge3/bin/activate

```

### 🧱 Create a project-specific conda environment

```bash
conda create --prefix ./env python=3.11
conda activate ./env

```

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/yashjaincodex/rag-evaluation-ragas-youtube
cd rag-evaluation-ragas-youtube

```

### 2. Create & activate environment

```bash
conda create --prefix ./env python=3.11
conda activate ./env

```

### 3. Install dependencies

```bash
pip install -r requirements.txt

```

### 4. Set your API key

```bash
cp .env.example .env
# Open .env and add your OpenAI API key

```

```env
OPENAI_API_KEY=sk-your-key-here

```

### 5. Run the demo

```bash
python rag_evaluation.py

```

---

## 📦 Dependencies

```
ragas==0.4.3
openai
python-dotenv

```

Install manually:

```bash
pip install ragas==0.4.3 openai python-dotenv

```

---

## 🧠 RAGAS v0.4 Metrics Explained

| Metric               | Needs Reference | What it detects                                                        |
| -------------------- | --------------- | ---------------------------------------------------------------------- |
| `Faithfulness`       | ❌              | **Hallucination** — did the LLM make up facts not in the context?      |
| `AnswerRelevancy`    | ❌              | **Off-topic response** — is the answer relevant to the question?       |
| `ContextRecall`      | ✅              | **Retrieval failure** — did the retriever fetch the right information? |
| `FactualCorrectness` | ✅              | **Wrong answer** — is the answer factually correct vs. the reference?  |

> ✅ = reference (ground truth) answer required  |  ❌ = works without ground truth

---

## 🎯 What the Demo Shows

The file `rag_evaluation.py` contains **5 carefully designed samples** to show how each metric catches a different type of failure:

| Sample      | Problem                                                   | Metric that catches it   |
| ----------- | --------------------------------------------------------- | ------------------------ |
| 🔴 Sample 1 | LLM added _"population of 5 million"_ — not in context    | `Faithfulness` LOW       |
| 🔴 Sample 2 | Retriever missed the Nobel Prize info                     | `ContextRecall` LOW      |
| 🔴 Sample 3 | LLM said 1901 instead of 1876                             | `FactualCorrectness` LOW |
| 🔴 Sample 4 | Asked about Python the language, answered about the snake | `AnswerRelevancy` LOW    |
| 🟢 Sample 5 | Perfect answer — baseline for comparison                  | All metrics HIGH         |

---

## 🔑 Key API Changes in RAGAS v0.4

If you've used RAGAS before, here's what changed:

|                        | v0.3 (old)                             | v0.4 (new)                                         |
| ---------------------- | -------------------------------------- | -------------------------------------------------- |
| **LLM setup**          | `LangchainLLMWrapper(ChatOpenAI(...))` | `llm_factory("gpt-4o-mini", client=AsyncOpenAI())` |
| **Embeddings**         | `LangchainEmbeddingsWrapper(...)`      | `OpenAIEmbeddings(client=client, model=...)`       |
| **Metrics import**     | `from ragas.metrics import ...`        | `from ragas.metrics.collections import ...`        |
| **Scoring**            | `evaluate(dataset, metrics, llm)`      | `await metric.ascore(**kwargs)`                    |
| **Return type**        | `float`                                | `MetricResult` → use `.value`                      |
| **Ground truth field** | `ground_truths=["answer"]`             | `reference="answer"`                               |

---

## 💡 Tips for Your YouTube Video

1. **Explain** `response` **vs** `reference` — response is what your LLM said, reference is the correct answer
2. **Walk through each bad sample** — show the low score and explain why RAGAS flagged it
3. **Highlight the good sample** — contrast it with the bad ones so viewers see the difference clearly
4. **Show the terminal output** — the ✅ ⚠️ ❌ emojis make it very visual and easy to follow
5. **Explain reference-free vs reference-based** — Faithfulness and AnswerRelevancy work in production without ground truth labels

---

## 🙌 Contributing

Pull requests are welcome! Please open an issue first for major changes.

## 📄 License

MIT © [Yash Jain](https://github.com/yashjaincodex)
