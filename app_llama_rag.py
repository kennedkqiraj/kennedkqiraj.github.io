# app_llama_rag.py
# Streamlit RAG over assets/projects.json and assets/further_training.json
# Uses Sentence-Transformers + NumPy cosine retrieval (no FAISS) + Groq Llama
# Tone: warm, confident, speaks in FIRST PERSON as Kened. No debug expander.

import os
import json
import re
import textwrap
from pathlib import Path
from typing import List, Dict, Any, Tuple

import streamlit as st
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# -------------------- App Config --------------------
st.set_page_config(
    page_title="I’m Kened’s assistant — how may I help you?",
    page_icon="💬",
    layout="centered",
)

# Optional: load JSON from raw GitHub URLs via env vars (otherwise local files)
PROJECTS_URL = os.getenv("PROJECTS_URL", "").strip() or None
TRAINING_URL = os.getenv("TRAINING_URL", "").strip() or None

LOCAL_CANDIDATES = [
    Path("assets"),
    Path(__file__).parent / "assets",
    Path("."),
]

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Groq hosted Llama
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

MAX_CTX_DOCS = 6
TEMPERATURE = 0.25
MAX_TOKENS = 550

# -------------------- Data helpers --------------------
def load_json_from(url: str | None, fallbacks: List[Path], filename: str) -> List[Dict[str, Any]]:
    """Load JSON from URL (if provided) else from local fallback paths."""
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
        [PROJECT] Title: {title} | Company/Brand: {brand}
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

# -------------------- Embeddings & Retrieval (NumPy) --------------------
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource(show_spinner=True)
def build_index(projects: List[Dict[str, Any]], training: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    """Returns (embedding_matrix, docs). Embeddings are L2-normalized."""
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
    emb_mat = embedder.encode(docs, show_progress_bar=False, normalize_embeddings=True)
    return emb_mat.astype("float32"), docs

def retrieve(query: str, emb_mat: np.ndarray, docs: List[str], k: int = MAX_CTX_DOCS) -> List[str]:
    embedder = get_embedder()
    qv = embedder.encode([query], normalize_embeddings=True).astype("float32")[0]
    scores = emb_mat @ qv  # cosine similarity since vectors normalized
    topk_idx = np.argsort(-scores)[:k]
    hits = [docs[i] for i in topk_idx]
    # dedupe while preserving order
    seen = set()
    out = []
    for h in hits:
        key = h[:160]
        if key not in seen:
            out.append(h)
            seen.add(key)
    return out

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
                            "You are Kened Kqiraj’s personal portfolio assistant. "
                            "You speak in FIRST PERSON as Kened (use 'I', 'me', 'my'). "
                            "Be warm, concise, and confident; professional with a light friendly tone. "
                            "Answer ONLY from the provided context (projects.json and further_training.json). "
                            "If something isn’t in context, say briefly that you don’t have that detail. "
                            "Do NOT mention being an AI or a chatbot. "
                            "Keep answers crisp; prefer bullet points for lists."
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
    Use only the context bullets below to answer.
    If the answer isn't in context, say you don't have that detail.
    Write as Kened in the first person.
    """).strip()
    ctx = "\n".join(f"- {c}" for c in contexts)
    user = f"Question: {question}\n\nContext:\n{ctx}\n\nAnswer:"
    return f"{system}\n\n{user}"

# -------------------- UI --------------------
# Stylish header (looks nicer in the popup)
st.markdown(
    """
    <div style="
      display:flex;align-items:center;gap:12px;
      padding:14px 16px;margin:8px 0 2px;border-radius:14px;
      background:linear-gradient(135deg, rgba(96,165,250,.18), rgba(110,231,183,.15));
      border:1px solid rgba(255,255,255,.08)
    ">
      <div style="font-size:24px">💬</div>
      <div>
        <div style="font-weight:800;font-size:20px;line-height:1.2">I’m Kened’s assistant — how may I help you?</div>
        <div style="color:#9aa3b2;font-size:13px">Groq Llama • Answers about my projects & training (from JSON)</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load JSON data (URL or local)
projects = load_json_from(PROJECTS_URL, LOCAL_CANDIDATES, "projects.json") or \
           load_json_from(PROJECTS_URL, LOCAL_CANDIDATES, "assets/projects.json")
training = load_json_from(TRAINING_URL, LOCAL_CANDIDATES, "further_training.json") or \
           load_json_from(TRAINING_URL, LOCAL_CANDIDATES, "assets/further_training.json")

if not projects and not training:
    st.warning("I can’t find my data (projects / further training). Please ensure both JSON files exist or set the URLs.")

EMB, DOCS = build_index(projects, training)

# Warm first message
if "history" not in st.session_state:
    st.session_state.history = [
        {"role": "assistant",
         "content": "Hi! I’m Kened’s assistant. Ask me about my projects, stack, experience, or training."}
    ]

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

q = st.chat_input("Type your question…")
if q:
    st.session_state.history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            ctxs = retrieve(q, EMB, DOCS, k=MAX_CTX_DOCS)
            prompt = build_prompt(q, ctxs)
            ans = llm_call(prompt)
            st.markdown(ans)
            st.session_state.history.append({"role": "assistant", "content": ans})
