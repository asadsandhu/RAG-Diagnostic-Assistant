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

iface = gr.Interface(
    fn=rag_chat,
    inputs=gr.Textbox(lines=3, placeholder="Enter a clinical query..."),
    outputs="text",
    title="🩺 Clinical Reasoning RAG Assistant",
    description="Ask a medical question based on MIMIC-IV-Ext-DiReCT's diagnostic knowledge.",
    allow_flagging="never"
)

iface.launch()
