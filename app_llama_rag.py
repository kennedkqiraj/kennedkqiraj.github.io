# app_llama_rag.py
# Streamlit + FAISS RAG over assets/projects.json and assets/further_training.json,
# using a hosted Llama via GROQ (free-tier friendly). No local Ollama needed.

import os
import json
import re
import textwrap
from pathlib import Path
from typing import List, Dict, Any, Tuple

import streamlit as st
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

# -------------------- Config --------------------
st.set_page_config(
    page_title="Kened • RAG Chatbot",
    page_icon="🦙",
    layout="centered",
)

# If you prefer to load JSONs from GitHub raw URLs, set these env vars to raw links.
# Otherwise, the app will load from local files in /assets.
PROJECTS_URL = os.getenv("PROJECTS_URL", "").strip() or None
TRAINING_URL = os.getenv("TRAINING_URL", "").strip() or None

LOCAL_CANDIDATES = [
    Path("assets"),
    Path(__file__).parent / "assets",
    Path("."),
]

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# GROQ hosted Llama
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Good default: fast + inexpensive
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

MAX_CTX_DOCS = 6
TEMPERATURE = 0.3
MAX_TOKENS = 450
SARCASM = True  # mild, professional

# -------------------- Data loading helpers --------------------
def load_json_from(url: str | None, fallbacks: List[Path], filename: str) -> List[Dict[str, Any]]:
    """Load JSON from a URL (if provided) or from local fallback paths."""
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

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += max(1, chunk_size - overlap)
    return chunks

def doc_from_project(p: Dict[str, Any]) -> str:
    title = p.get("title", "")
    brand = p.get("brand", "")
    desc = p.get("desc", "")
    bullets = " • ".join(p.get("bullets", []))
    tags = ", ".join(p.get("tags", []))
    return normalize_space(f"""
        [PROJECT] Title: {title} | Brand: {brand}
        Description: {desc}
        Bullets: {bullets}
        Tags: {tags}
    """)

def doc_from_training(t: Dict[str, Any]) -> str:
    org = t.get("org", "")
    title = t.get("title", "")
    period = t.get("period", "")
    bullets = " • ".join(t.get("bullets", []))
    tags = ", ".join(t.get("tags", []))
    return normalize_space(f"""
        [TRAINING] {title} — {org} ({period})
        Details: {bullets}
        Tags: {tags}
    """)

# -------------------- Embeddings & Index --------------------
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource(show_spinner=True)
def build_index(projects: List[Dict[str, Any]], training: List[Dict[str, Any]]) -> Tuple[faiss.IndexFlatIP, List[str]]:
    docs: List[str] = []

    # Projects
    for p in projects:
        base = doc_from_project(p)
        for ch in chunk_text(base, 120, 20):
            if ch:
                docs.append(ch)

    # Training
    for t in training:
        base = doc_from_training(t)
        for ch in chunk_text(base, 120, 20):
            if ch:
                docs.append(ch)

    if not docs:
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
    # dedupe while preserving order
    seen = set()
    uniq = []
    for h in hits:
        key = h[:160]
        if key not in seen:
            uniq.append(h)
            seen.add(key)
    return uniq

# -------------------- LLM (Groq) --------------------
def llm_call(prompt: str) -> str:
    if not GROQ_API_KEY:
        return ("Hosted LLM missing: set GROQ_API_KEY secret in Streamlit Cloud.\n"
                "App → Settings → Secrets → GROQ_API_KEY=...")

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are Kened Kqiraj's portfolio assistant. "
                            "Answer ONLY using the provided context. "
                            "Be concise, specific, and professional. "
                            "A tiny, tasteful hint of sarcasm is okay; avoid snark. "
                            "If unknown from context, say so briefly."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"LLM error: {e}"

def build_prompt(question: str, contexts: List[str]) -> str:
    system = textwrap.dedent("""
    Use the context bullets below to answer the user's question.
    Do NOT invent facts. If the answer isn't in context, say you don't have that detail.
    """).strip()
    ctx = "\n".join(f"- {c}" for c in contexts)
    user = f"Question: {question}\n\nContext:\n{ctx}\n\nAnswer:"
    return f"{system}\n\n{user}"

# -------------------- UI --------------------
st.title("🦙 RAG Chatbot")
st.caption("Groq Llama • Answers about projects & further training from JSON.")

# Load JSON data (URL or local)
projects = load_json_from(PROJECTS_URL, LOCAL_CANDIDATES, "projects.json") or \
           load_json_from(PROJECTS_URL, LOCAL_CANDIDATES, "assets/projects.json")
training = load_json_from(TRAINING_URL, LOCAL_CANDIDATES, "further_training.json") or \
           load_json_from(TRAINING_URL, LOCAL_CANDIDATES, "assets/further_training.json")

if not projects and not training:
    st.warning("Could not load JSON data. Ensure `assets/projects.json` and `assets/further_training.json` exist, "
               "or set PROJECTS_URL/TRAINING_URL to raw GitHub URLs.")

index, DOCS = build_index(projects, training)

# Friendly first message
if "history" not in st.session_state:
    st.session_state.history = [
        {
            "role": "assistant",
            "content": "Here is the chat bot that I created to write on my behalf — what would you like to know?",
        }
    ]

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

q = st.chat_input("Ask about a project, stack, or training…")
if q:
    st.session_state.history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            ctxs = retrieve(q, index, DOCS, k=MAX_CTX_DOCS)
            prompt = build_prompt(q, ctxs)
            ans = llm_call(prompt)
            if SARCASM and ans and "LLM error" not in ans and "missing" not in ans.lower():
                ans += "\n\n*There you go—just the essentials.*"
            st.markdown(ans)
            st.session_state.history.append({"role": "assistant", "content": ans})

# Optional: small debug panel
with st.expander("🔍 RAG context (debug)"):
    st.write("First few doc chunks:", DOCS[:5])
    st.write(
        {
            "projects_loaded": len(projects),
            "training_loaded": len(training),
            "model": GROQ_MODEL,
        }
    )
