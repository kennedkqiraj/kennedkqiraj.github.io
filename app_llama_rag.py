# app_llama_rag.py
# Streamlit RAG over assets/projects.json and assets/further_training.json
# Uses Sentence-Transformers + NumPy cosine retrieval (no FAISS) + Groq Llama
# HR-aware: deflects sensitive/negotiation questions with a friendly CTA.
# Voice: warm, confident, persuasive, FIRST PERSON (I / me / my).

import os
import json
import re
import textwrap
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# -------------------- App & Profile Config --------------------
st.set_page_config(
    page_title="I’m Kened’s assistant — how may I help you?",
    page_icon="💬",
    layout="centered",
)

# Contact / CTA configuration (edit these!)
MY_PHONE = os.getenv("MY_PHONE", "+41 76 567 29 85")
MY_EMAIL = os.getenv("MY_EMAIL", "kened.kqiraj@stud.hslu.ch")

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

# Optional profile facts
PROFILE = {
    "location": "Zug, Switzerland",
    "roles_open": ["Data Scientist", "AI Engineer", "Data Analyst"],
    "work_mode": ["On-site", "Hybrid", "Remote"],
    "status": "Open to full-time roles",
    "start_timing": "Flexible start; currently interning at On AG.",
    "languages": ["English", "German (Intermediate)", "Albanian"],
}

# -------------------- Data helpers --------------------
def load_json_from(url: Optional[str], fallbacks: List[Path], filename: str) -> List[Dict[str, Any]]:
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

    for p in projects:
        base = doc_from_project(p)
        for ch in chunk_text(base, 120, 20):
            if ch:
                docs.append(ch)

    for t in training:
        base = doc_from_training(t)
        for ch in
