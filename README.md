# 🩺 RAGnosis – Clinical Reasoning via Retrieval-Augmented Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-RAGnosis-blue?logo=huggingface)](https://huggingface.co/spaces/asadsandhu/RAGnosis)
[![GitHub Repo](https://img.shields.io/badge/GitHub-asadsandhu/RAG--Diagnostic--Assistant-black?logo=github)](https://github.com/asadsandhu/RAG-Diagnostic-Assistant)

> ⚕️ A lightweight Retrieval-Augmented Generation (RAG) assistant for clinical diagnosis, trained on annotated medical notes from **MIMIC-IV-Ext-DiReCT**, and deployable both on **GPU (fast)** and **CPU (slow)** modes.

---

## 🖼️ Demo

🎯 Try the model live (CPU deployment):  
🔗 [RAGnosis – Hugging Face Spaces](https://huggingface.co/spaces/asadsandhu/RAGnosis)

<p align="center">
  <img src="assets/demo.png" alt="Demo" width="750">
</p>

---

## ⚙️ Tech Stack

| Layer        | Details                                                                 |
|--------------|-------------------------------------------------------------------------|
| 🧠 Model      | [`Nous-Hermes-2-Mistral-7B-DPO`](https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO) (GPU) / [`BioMistral-7B`](https://huggingface.co/BioMistral/BioMistral-7B) (CPU) |
| 🏥 Dataset    | [`MIMIC-IV-Ext-DiReCT`](https://github.com/asadsandhu/RAG-Diagnostic-Assistant/blob/main/mimic-iv-ext-direct-1.0.0.zip)              |
| 🔍 Retriever  | FAISS + SentenceTransformers (`all-MiniLM-L6-v2`)                      |
| 💻 Frontend   | Gradio (via Hugging Face Spaces)                                       |
| 🧠 Backend    | PyTorch + Transformers + BitsAndBytes                                  |

---

## 🚀 Features

- 🔎 Top-k retrieval from real, annotated clinical notes
- 🧠 Explainable diagnosis using structured logic and LLMs
- 📋 Based on real diagnostic chains from MIMIC-IV-Ext-DiReCT
- 💬 Clean Gradio UI for free-text medical queries
- ✅ Supports GPU for fast inference or CPU fallback

---

## 🧪 Pipeline Overview

### ✅ Step 1: Preprocessing

- Parse annotated `.json` samples and knowledge graphs
- Chunk clinical facts into `retrieval_corpus.csv`
- Embed chunks using Sentence-BERT
- Save embeddings into `faiss_index.bin`

### ✅ Step 2: Retrieval (FAISS)

- Query is embedded using MiniLM
- Top-k chunks are returned using FAISS index

### ✅ Step 3: Generation

- Query and context are merged into a prompt
- Model (Mistral-7B) generates the diagnosis
- Output is parsed and shown in Gradio UI

---

## ⚠️ Deployment Modes (CPU vs GPU)

| Feature      | Hugging Face (CPU)          | Local/Colab (GPU)                      |
|--------------|-----------------------------|----------------------------------------|
| Model Used   | `BioMistral/BioMistral-7B`   | `Nous-Hermes-2-Mistral-7B-DPO`         |
| Speed        | 🐢 ~500 seconds per query    | ⚡ <10 seconds per query                |
| Accuracy     | ✅ Good                      | ✅ Great (instruction-tuned)            |
| Setup        | Ready-to-use (slow)         | Requires CUDA but runs super fast      |
| Hosting      | Free (Hugging Face Spaces)  | Free (Colab, Kaggle, local CUDA)       |

> 💡 For real-time use, prefer running the CUDA version via GitHub clone. Hugging Face version is for preview/demo only.

---

## 🛠 Run Locally with GPU (Recommended)

### 🔁 1. Clone the Repository

```bash
git clone https://github.com/asadsandhu/RAG-Diagnostic-Assistant.git
cd RAG-Diagnostic-Assistant
````

### 📦 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 🚀 3. Run the App (CUDA Required)

```bash
python app.py
```

✔️ Required files are included:

* `retrieval_corpus.csv`
* `faiss_index.bin`

---

## 📁 Folder Structure

```
RAG-Diagnostic-Assistant/
├── app.py                  # Deployable backend using Gradio
├── RAGnosis.ipynb          # Notebook version of the pipeline
├── faiss_index.bin         # FAISS vector index
├── retrieval_corpus.csv    # Processed clinical chunks
├── requirements.txt        # Dependencies
├── assets/
│   └── demo.png            # Sample UI screenshot
└── README.md
```

---

## 📚 Dataset: MIMIC-IV-Ext-DiReCT

* Combines annotated diagnostic chains (in `samples/`) and structured graphs (in `diagnostic_kg/`)
* Captures how clinicians move from **symptom → rationale → diagnosis**
* Original repo: [DiReCT GitHub](https://github.com/asadsandhu/RAG-Diagnostic-Assistant/blob/main/mimic-iv-ext-direct-1.0.0.zip)

---

## ⚡ Sample Prompt

> Query:
> "patient is experiencing shortness of breath"

💬 **LLM Output**:

> "Shortness of breath is a common symptom that can be caused by a variety of respiratory conditions. The differential diagnosis for shortness of breath includes asthma, chronic obstructive pulmonary disease (COPD), congestive heart failure, pneumonia, and pneumothorax. In order to determine the cause of the shortness of breath, it is important to consider the patient's medical history, physical examination findings, and diagnostic testing results. For example, if the patient has a history of asthma and is experiencing wheezing and a prolonged expiratory phase on examination, this would suggest asthma as the cause of the shortness of breath. On the other hand, if the patient has a history of congestive heart failure and is experiencing orthopnea, crackles on auscultation, and a history of edema, this would suggest congestive heart failure as the cause of the shortness of breath."

---

## 📣 Medium Blog

📖 Read the full blog explaining RAGnosis, dataset structure, pipeline design, and tradeoffs:

👉 [Read on Medium](https://medium.com/@asadsandhu/ragnosis-building-a-clinical-diagnostic-assistant-with-retrieval-augmented-generation-39093bdf7dd4)

---

## 👤 Author & Links

Built by **Asad Ali**
AI Developer & NLP Researcher

* 🔗 [LinkedIn](https://www.linkedin.com/in/asadsandhu0/)
* 🧠 [Medium](https://medium.com/@asadsandhu)
* 💻 [GitHub](https://github.com/asadsandhu)
* 🤗 [Hugging Face](https://huggingface.co/asadsandhu)

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

* 🏥 MIT-LCP for MIMIC-IV dataset
* 🧪 DiReCT team for annotated clinical reasoning data
* 🤗 Hugging Face Transformers & Gradio
* 🔍 Facebook Research for FAISS
* 🧠 Nous Research for Mistral models

---

> ⚠️ *Disclaimer: This tool is for academic and demo purposes only. Not intended for clinical use.*
