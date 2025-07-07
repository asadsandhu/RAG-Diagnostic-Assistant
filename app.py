import gradio as gr
import pandas as pd
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

# ----------------------
# Load Retrieval Corpus & FAISS Index
# ----------------------
df = pd.read_csv("retrieval_corpus.csv")
index = faiss.read_index("faiss_index.bin")

# ----------------------
# Load Embedding Model
# ----------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------
# Load HuggingFace LLM (Nous-Hermes)
# ----------------------
model_id = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
generation_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=bnb_config
)

# ----------------------
# RAG Functions
# ----------------------

def retrieve_top_k(query, k=5):
    query_embedding = embedding_model.encode([query]).astype("float32")
    D, I = index.search(query_embedding, k)
    results = df.iloc[I[0]].copy()
    results["score"] = D[0]
    return results

def build_prompt(query, retrieved_docs):
    context_text = "\n".join([
        f"- {doc['text']}" for _, doc in retrieved_docs.iterrows()
    ])

    prompt = f"""[INST] <<SYS>>
You are a medical assistant trained on clinical reasoning data. Given the following patient query and related clinical observations, generate a diagnostic explanation or suggestion based on the context.
<</SYS>>

### Patient Query:
{query}

### Clinical Context:
{context_text}

### Diagnostic Explanation:
[/INST]
"""
    return prompt

def generate_local_answer(prompt, max_new_tokens=512):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    output = generation_model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.5,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split("### Diagnostic Explanation:")[-1].strip()

# ----------------------
# Gradio Interface
# ----------------------

def rag_chat(query):
    top_docs = retrieve_top_k(query, k=5)
    prompt = build_prompt(query, top_docs)
    answer = generate_local_answer(prompt)
    return answer

# Optional: basic CSS to enhance layout
custom_css = """
textarea, .input_textbox {
    font-size: 1.05rem !important;
}
.output-markdown {
    font-size: 1.08rem !important;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Default(primary_hue="blue")) as demo:
    gr.Markdown("""
# 🩺 RAGnosis — Clinical Reasoning Assistant

Enter a natural-language query describing your patient's condition to receive an AI-generated diagnostic reasoning response.

**Example:**  
*Patient has shortness of breath, fatigue, and leg swelling.*
""")

    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(
                lines=4,
                label="📝 Patient Query",
                placeholder="Enter patient symptoms or findings..."
            )
            submit_btn = gr.Button("🔍 Generate Diagnosis")

        with gr.Column():
            output = gr.Markdown(label="🧠 Diagnostic Reasoning")

    submit_btn.click(fn=rag_chat, inputs=query_input, outputs=output)

demo.launch(share=True)
