# app_llama_rag.py
# Chatbot that uses RAG over assets/projects.json AND assets/further_training.json
# Requires: streamlit, sentence-transformers, faiss-cpu, requests (for Ollama)
# Run:  streamlit run app_llama_rag.py
# Make sure an Ollama Llama model is available, e.g.:
#   ollama pull llama3.1:8b
#   (or adjust OLLAMA_MODEL below)

import os, json, re, textwrap, time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import streamlit as st
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

# -------------------- Config --------------------
st.set_page_config(page_title="Kened • RAG Chatbot", page_icon="🦙", layout="centered")

# Use local files or raw GitHub URLs
PROJECTS_URL: str | None = None  # e.g. "https://raw.githubusercontent.com/kennedkqiraj/kennedkqiraj.github.io/main/assets/projects.json"
TRAINING_URL: str | None = None  # e.g. "https://raw.githubusercontent.com/kennedkqiraj/kennedkqiraj.github.io/main/assets/further_training.json"

LOCAL_CANDIDATES = [
    Path("assets"),
    Path(__file__).parent / "assets",
    Path(".")
]

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")  # change if you prefer a different size

MAX_CTX_DOCS = 6
MAX_TOKENS = 450  # approximate for small answers; Ollama ignores if too high
TEMPERATURE = 0.3

# Mild sarcasm toggle
SARCASM = True

# -------------------- Helpers --------------------
def load_json_from(url: str | None, fallbacks: List[Path], filename: str) -> List[Dict[str, Any]]:
    if url:
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=10) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            pass
    for base in fallbacks:
        p = base / filename
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return []

@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += (chunk_size - overlap)
    return chunks

def doc_from_project(p: Dict[str, Any]) -> str:
    title = p.get("title","")
    brand = p.get("brand","")
    desc = p.get("desc","")
    bullets = " • ".join(p.get("bullets", []))
    tags = ", ".join(p.get("tags", []))
    return normalize_space(f"""
        [PROJECT] Title: {title} | Brand: {brand}
        Description: {desc}
        Bullets: {bullets}
        Tags: {tags}
    """)

def doc_from_training(t: Dict[str, Any]) -> str:
    org = t.get("org","")
    title = t.get("title","")
    period = t.get("period","")
    bullets = " • ".join(t.get("bullets", []))
    tags = ", ".join(t.get("tags", []))
    return normalize_space(f"""
        [TRAINING] {title} — {org} ({period})
        Details: {bullets}
        Tags: {tags}
    """)

@st.cache_resource(show_spinner=False)
def build_index(projects: List[Dict[str,Any]], training: List[Dict[str,Any]]) -> Tuple[faiss.IndexFlatIP, List[str]]:
    docs: List[str] = []
    # flatten projects
    for p in projects:
        base = doc_from_project(p)
        for ch in chunk_text(base, 120, 20):  # small logical chunks; content is short anyway
            docs.append(ch)
    # flatten training
    for t in training:
        base = doc_from_training(t)
        for ch in chunk_text(base, 120, 20):
            docs.append(ch)

    if not docs:
        # create a dummy entry to avoid FAISS errors
        docs = ["No documents loaded."]

    embedder = get_embedder()
    vecs = embedder.encode(docs, show_progress_bar=False, normalize_embeddings=True)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(np.array(vecs, dtype="float32"))
    return index, docs

def retrieve(query: str, index: faiss.IndexFlatIP, docs: List[str], k: int = MAX_CTX_DOCS) -> List[str]:
    embedder = get_embedder()
    qv = embedder.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(qv, dtype="float32"), k)
    hits = [docs[i] for i in I[0] if 0 <= i < len(docs)]
    # dedupe lightly while preserving order
    seen = set()
    uniq = []
    for h in hits:
        key = h[:120]
        if key not in seen:
            uniq.append(h)
            seen.add(key)
    return uniq

def llm_call(prompt: str) -> str:
    # Ollama REST API
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "temperature": TEMPERATURE,
        "num_predict": MAX_TOKENS,
    }
    # stream response
    r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, stream=True, timeout=120)
    r.raise_for_status()
    out = []
    for line in r.iter_lines():
        if not line:
            continue
        try:
            obj = json.loads(line.decode("utf-8"))
        except Exception:
            continue
        if "response" in obj:
            out.append(obj["response"])
    return "".join(out).strip()

def build_prompt(question: str, contexts: List[str]) -> str:
    system = textwrap.dedent("""
    You are Kened Kqiraj's portfolio assistant. Answer using ONLY the provided context.
    Be concise, specific, and helpful. It's okay to include a tiny bit of playful sarcasm,
    but keep it professional and subtle. If something is unknown, say so briefly.
    Then stop.
    """).strip()

    ctx = "\n\n".join(f"- {c}" for c in contexts)
    user = f"Question: {question}\n\nContext:\n{ctx}\n\nAnswer:"

    return f"{system}\n\n{user}"

# -------------------- UI --------------------
st.title("🦙 RAG Chatbot")
st.caption("Backed by Llama via Ollama • Answers about projects & further training (JSON).")

# Load data
projects = load_json_from(PROJECTS_URL, LOCAL_CANDIDATES, "projects.json") or \
           load_json_from(PROJECTS_URL, LOCAL_CANDIDATES, "assets/projects.json")
training = load_json_from(TRAINING_URL, LOCAL_CANDIDATES, "further_training.json") or \
           load_json_from(TRAINING_URL, LOCAL_CANDIDATES, "assets/further_training.json")

index, DOCS = build_index(projects, training)

# Chat
if "history" not in st.session_state:
    st.session_state.history = [
        {"role":"assistant","content":"Here is the chat bot that I created to write on my behalf — what would you like to know?"}
    ]

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

q = st.chat_input("Ask about a project, stack, or training…")
if q:
    st.session_state.history.append({"role":"user","content":q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking (summoning the llama)…"):
            ctxs = retrieve(q, index, DOCS, k=MAX_CTX_DOCS)
            prompt = build_prompt(q, ctxs)
            try:
                ans = llm_call(prompt)
            except Exception as e:
                ans = f"LLM error: {e}\n\nMake sure Ollama is running and the model `{OLLAMA_MODEL}` is pulled."
        if SARCASM and ans:
            ans += "\n\n*There you go—no fluff, just the good bits.*"
        st.markdown(ans)
        st.session_state.history.append({"role":"assistant","content":ans})

# Debug expander (optional)
with st.expander("🔍 RAG context (debug)"):
    st.write(DOCS[:5])
