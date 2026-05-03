# 🔍 RAG Evaluation with RAGAS + Streamlit

A full-stack application to evaluate RAG (Retrieval-Augmented Generation) pipelines
using the [RAGAS](https://docs.ragas.io) framework, with an interactive Streamlit UI.

> 📺 **[YouTube Tutorial](https://www.youtube.com/@YashJainCodex)** · 🐙 **[GitHub Repo](https://github.com/yashjaincodex/rag-evaluation-ragas-youtube)**

---

## 📁 Project Structure

```
rag-ragas-eval/
├── app.py              # Streamlit UI (3 tabs: Chat, Batch, History)
├── rag_pipeline.py     # LangChain + FAISS RAG pipeline
├── evaluator.py        # RAGAS evaluation logic
├── main.py             # CLI runner (no UI)
├── requirements.txt    # Python dependencies
├── .env.example        # API key template
└── README.md
```

---

## 🐍 Install Python Using Miniconda / Miniforge

To keep your AI projects clean and organized, it is recommended to use **conda environments**.

### 🔗 Download Miniforge for macOS (ARM64)

Download from the official repository:

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
conda create --prefix ./env python=3.13
conda activate ./env
```

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/yashjaincodex/rag-evaluation-ragas-youtube
cd rag-evaluation-ragas-youtube
```

### 2. Create & activate environment (if not done above)

```bash
conda create --prefix ./env python=3.13
conda activate ./env
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your API key

```bash
cp .env.example .env
# Open .env and paste your OpenAI API key
```

### 5a. Run the Streamlit UI

```bash
streamlit run app.py
```

### 5b. Run the CLI evaluator (no UI)

```bash
python main.py
```

---

## 🧠 RAGAS Metrics Explained

| Metric                  | Requires Reference | What it measures                                     |
| ----------------------- | ------------------ | ---------------------------------------------------- |
| **Faithfulness**        | ❌                 | Is the answer grounded in retrieved context?         |
| **Response Relevancy**  | ❌                 | Is the answer relevant to the question?              |
| **Context Precision**   | ✅                 | Is the retrieved context relevant to the query?      |
| **Context Recall**      | ✅                 | Does the retrieved context contain the ground truth? |
| **Factual Correctness** | ✅                 | Is the answer factually correct vs. the reference?   |

> ✅ = reference answer required &nbsp;|&nbsp; ❌ = works without ground truth

---

## 🖥️ Streamlit UI Features

### Tab 1 — 💬 Chat & Evaluate

- Live chat with your RAG pipeline
- RAGAS scores shown after every response (bar + radar charts)
- View retrieved context chunks
- Optional ground-truth input for reference-based metrics

### Tab 2 — 📦 Batch Evaluation

- Paste multiple Q&A pairs (`question|reference` format)
- Run bulk evaluation with progress bar
- Color-coded heatmap results table
- Download results as CSV

### Tab 3 — 📊 History & Analytics

- Track scores across all chat turns
- Trend line chart over time
- Per-turn drill-down with radar chart
- Download full history as CSV

---

## 📦 Key Packages

| Package            | Version | Purpose                    |
| ------------------ | ------- | -------------------------- |
| `ragas`            | ≥0.2    | RAG evaluation metrics     |
| `langchain`        | ≥0.3    | RAG pipeline framework     |
| `langchain-openai` | ≥0.2    | OpenAI LLM + embeddings    |
| `faiss-cpu`        | ≥1.8    | Vector store for retrieval |
| `streamlit`        | ≥1.40   | Web UI                     |
| `plotly`           | ≥5.0    | Charts and visualizations  |

---

## 💡 Tips for Your YouTube Video

1. **Walk through `rag_pipeline.py` first** — show how documents become chunks → embeddings → FAISS index
2. **Explain `SingleTurnSample`** — the new RAGAS 0.2 API vs. the old HuggingFace dict approach
3. **Show the live UI** — chat → watch scores appear → explain each metric
4. **Use the batch tab** — paste 5 questions and show the heatmap
5. **History tab** — show the trend line dropping or improving as questions get harder

---

## 🙌 Contributing

Pull requests are welcome! Please open an issue first for major changes.

## 📄 License

MIT © [Yash Jain](https://github.com/yashjaincodex)
