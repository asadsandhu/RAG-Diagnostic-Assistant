```markdown
<h1 align="center">🩺 RAGnosis</h1>
<h3 align="center">Clinical Reasoning Assistant using MIMIC-IV-Ext-DiReCT & RAG</h3>

<p align="center">
  <img src="assets/demo.png" alt="RAGnosis Demo" width="750"/>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/asadsandhu/RAGnosis"><img alt="HF Space" src="https://img.shields.io/badge/Try%20Live%20App-%F0%9F%94%8D%20RAGnosis-blue?style=for-the-badge&logo=gradio"></a>
  <a href="https://github.com/asadsandhu/RAG-Diagnostic-Assistant"><img alt="GitHub Repo" src="https://img.shields.io/badge/View%20Code-%F0%9F%92%BB%20GitHub-black?style=for-the-badge&logo=github"></a>
</p>

---

## 🧠 What is RAGnosis?

**RAGnosis** is an LLM-powered **diagnostic reasoning assistant** that uses [MIMIC-IV-Ext-DiReCT](https://github.com/asadsandhu/RAG-Diagnostic-Assistant/blob/main/mimic-iv-ext-direct-1.0.0.zip) — a dataset of real-world ICU clinical notes and annotated diagnostic trees — to generate accurate and explainable answers for clinicians and researchers.

It combines **FAISS retrieval**, **clinical knowledge graphs**, and **Generative LLMs (Mistral 7B)** in a Retrieval-Augmented Generation (RAG) pipeline.

---

## 🔥 Try it Live

🚀 **Launch the app here**:  
🔗 [https://huggingface.co/spaces/asadsandhu/RAGnosis](https://huggingface.co/spaces/asadsandhu/RAGnosis)

---

## ⚙️ Tech Stack

| Layer | Component |
|-------|-----------|
| 🧠 LLM | [`Nous-Hermes-2-Mistral-7B-DPO`](https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO) |
| 📖 Dataset | [MIMIC-IV-Ext-DiReCT](https://github.com/asadsandhu/RAG-Diagnostic-Assistant/blob/main/mimic-iv-ext-direct-1.0.0.zip) |
| 🔍 Retriever | FAISS + `all-MiniLM-L6-v2` (Sentence Transformers) |
| 🧾 Backend | Python + Transformers + BitsAndBytes |
| 💻 Interface | [Gradio](https://gradio.app/) (on Hugging Face Spaces) |

---

## 🩺 Features

- 🔎 Top-k retrieval from annotated notes + clinical KG
- 💡 Reasoning powered by local Mistral 7B model (offline-capable)
- 📄 Diagnoses explained step-by-step in plain English
- 📊 Interactive Gradio interface with natural query input
- 🧾 Fully open-source with fast FAISS-based search

---

## 🧪 Example Query

```

Patient shows signs of edema, orthopnea, and fatigue.

````

💬 RAGnosis Response:
> The most likely diagnosis is **congestive heart failure (CHF)**. This is indicated by the symptoms of edema (fluid buildup), orthopnea (difficulty breathing while lying down), and fatigue due to decreased cardiac output...

---

## 🧰 How It Works

### 🔹 Step 1: Preprocessing
- Unzip and parse `samples/` and `diagnostic_kg/`
- Flatten diagnostic trees, annotated nodes, and observations
- Generate chunks for retrieval

### 🔹 Step 2: Vector Retrieval
- Embed chunks using `all-MiniLM-L6-v2`
- Build FAISS index → [faiss_index.bin](https://github.com/asadsandhu/RAG-Diagnostic-Assistant/blob/main/faiss_index.bin)
- Store metadata → [retrieval_corpus.csv](https://github.com/asadsandhu/RAG-Diagnostic-Assistant/blob/main/retrieval_corpus.csv)

### 🔹 Step 3: Generation
- Top-k chunks are inserted into an `[INST]`-style prompt
- Model: `Nous-Hermes-2-Mistral-7B-DPO`
- Response is decoded and returned to the user

---

## 🏁 Run Locally

```bash
# Clone the repo
git clone https://github.com/asadsandhu/RAG-Diagnostic-Assistant.git
cd RAG-Diagnostic-Assistant

# Install dependencies
pip install -r requirements.txt

# Run the Gradio app
python app.py
````

Make sure to download and place these in the root folder:

* ✅ [faiss\_index.bin](https://github.com/asadsandhu/RAG-Diagnostic-Assistant/blob/main/faiss_index.bin)
* ✅ [retrieval\_corpus.csv](https://github.com/asadsandhu/RAG-Diagnostic-Assistant/blob/main/retrieval_corpus.csv)

---

## 📂 Project Structure

```
RAG-Diagnostic-Assistant/
├── app.py
├── faiss_index.bin
├── retrieval_corpus.csv
├── requirements.txt
├── assets/
│   └── demo.png
└── README.md
```

---

## 👨‍💻 Author

Built with ❤️ by [Asad Ali](https://www.linkedin.com/in/asadsandhu0)
🔗 GitHub: [@asadsandhu](https://github.com/asadsandhu)
🔗 Hugging Face: [RAGnosis Space](https://huggingface.co/spaces/asadsandhu/RAGnosis)

---

## 📄 License

Licensed under the **MIT License** — free for personal, academic, or commercial use with attribution.

---

## 🙏 Acknowledgments

* 🏥 Dataset: [MIMIC-IV-Ext-DiReCT](https://github.com/wbw520/DiReCT)
* 🧠 Model: [Nous-Hermes-2 Mistral-7B-DPO](https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO)
* 🔍 Retrieval: [FAISS](https://github.com/facebookresearch/faiss)
* 🌐 Deployment: [Gradio](https://gradio.app/), [Hugging Face Spaces](https://huggingface.co/spaces)

---

> ⚠️ **Disclaimer**: This project is for educational and research purposes only. It is not intended to provide medical advice, diagnosis, or treatment.
