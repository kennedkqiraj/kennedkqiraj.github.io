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

# -------------------- HR Guardrail & CTA --------------------
HR_SENSITIVE_PATTERNS = [
    r"\b(salary|compensation|pay|rate|wage|stipend|hourly|day rate|expected pay|desired salary|range)\b",
    r"\b(equity|bonus|stock|rsu|options)\b",
    r"\b(contract|notice period|termination|non[- ]?compete|non[- ]?solicit|probation)\b",
    r"\b(visa|sponsor|sponsorship|work authorization|work permit)\b",
    r"\b(relocate|relocation|move countries?)\b",
    r"\b(benefits?|pension|healthcare|insurance|vacation|holidays?)\b",
]

def hr_guardrail(user_text: str) -> Optional[str]:
    q = user_text.lower()
    if any(re.search(p, q) for p in HR_SENSITIVE_PATTERNS):
        return (
            f"I’d love to discuss that live so we align quickly on role scope and expectations. "
            f"Please call me at **{MY_PHONE}**\n\n"
            f"If it helps, feel free to email me at **{MY_EMAIL}** with a couple of time slots."
        )
    return None

# -------------------- Fun/Sarcasm Guardrail: "10 years" etc. -----------------
FUTURE_Q_PATTERNS = [
    r"\bwhere do you see yourself\b",
    r"\b10\s*years\b",
    r"\bfive\s*years\b",
    r"\b5\s*years\b",
    r"\bten\s*years\b",
    r"\blong[- ]?term (plan|goal|vision)s?\b",
    r"\bcareer (aspiration|goal)s?\b",
    r"\bfuture (plan|goal|vision)s?\b",
]

def is_future_projection_query(q: str) -> bool:
    ql = q.lower()
    return any(re.search(p, ql) for p in FUTURE_Q_PATTERNS)

def sarcastic_future_reply() -> str:
    # respectful, playful, and gives no real info
    lines = [
        "Ah, the crystal-ball classic. In ten years I’ll be exactly ten years older, which so far is the most accurate forecast on record.",
        "My long-term plan is simple: keep learning fast and shipping value. Specific prophecies are under NDA with Future-Me.",
        "Let’s trade horoscopes for roadmaps—ask me about shipped work or outcomes and I’ll happily dive in.",
    ]
    # Join them as a short quip + two light bullets
    return (
        f"{lines[0]}\n\n"
        f"- {lines[1]}\n"
        f"- {lines[2]}"
    )

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
                            "Your goal is to help the visitor understand why I’m a strong hire. "
                            "Be warm, concise, and confident; professional with a friendly tone. "
                            "A touch of light, good-natured sarcasm is welcome when appropriate—"
                            "keep it respectful and never hostile. "
                            "Answer ONLY from the provided context (projects.json and further_training.json) "
                            "and the short profile facts I provide in this message:\n\n"
                            f"PROFILE FACTS: {PROFILE}\n\n"
                            "Important: If a question asks about salary, compensation, sponsorship, contract terms, "
                            "benefits, or similar negotiation topics, DO NOT provide specifics. "
                            "Instead, politely ask them to call me or send me an E-Mail.\n"
                            "Prefer bullet points for lists; highlight impact and results."
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
    Use only the context bullets below (plus the short profile facts) to answer.
    If the answer isn't in context, say you don't have that detail.
    Write as Kened in the first person.
    Keep the answer focused and persuasive for hiring.
    """).strip()
    ctx = "\n".join(f"- {c}" for c in contexts)
    user = f"Question: {question}\n\nContext:\n{ctx}\n\nAnswer:"
    return f"{system}\n\n{user}"

# >>> PRESENT PROJECT LOGIC ----------------------------------------------------
def is_present_projects_query(q: str) -> bool:
    ql = q.lower()
    patterns = [
        r"\bpresent project(s)?\b",
        r"\bcurrent project(s)?\b",
        r"\bongoing project(s)?\b",
        r"\bwhat (are|am) you working on\b",
        r"\bworking on now\b",
        r"\bnowadays\b",
    ]
    return any(re.search(p, ql) for p in patterns)

def present_projects_only(projects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for p in projects:
        title = (p.get("title") or "").lower()
        if "present" in title:
            out.append(p)
    return out

def build_present_prompt(question: str, present_ps: List[Dict[str, Any]]) -> str:
    contexts = [doc_from_project(p) for p in present_ps]
    system = textwrap.dedent("""
    You are Keneds assistant describing ONLY the projects whose title contains the word "Present".
    Make it engaging and concise for a hiring audience:
    - Start with a one-sentence hook (impact/outcome).
    - Then give 3–5 punchy bullets: what I'm building, stack, why it matters.
    - Keep it first-person, confident, concrete.
    Do NOT mention other projects unless explicitly asked.
    If nothing is available, say I don't have a project marked as present.
    """).strip()
    ctx = "\n".join(f"- {c}" for c in contexts) if contexts else "- (no present projects found)"
    user = f"Question: {question}\n\nContext (only 'Present' projects):\n{ctx}\n\nAnswer:"
    return f"{system}\n\n{user}"
# -----------------------------------------------------------------------------

# --- QA JSON LOGGING ----------------------------------------------------------
from datetime import datetime, timezone
import uuid

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

def _qa_path() -> Path:
    """
    Prefer assets/qa.json next to the app.
    Falls back to ./qa.json if assets doesn't exist.
    """
    for base in LOCAL_CANDIDATES:
        try:
            base.mkdir(parents=True, exist_ok=True)
            if base.name == "assets":
                return base / "qa.json"
        except Exception:
            continue
    return Path("./qa.json")

def _load_qa(path: Path) -> list:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []

def _save_qa(path: Path, qa_list: list):
    try:
        path.write_text(json.dumps(qa_list, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # Don't crash chat if file is read-only or storage is ephemeral
        pass

def log_qa(question: str, answer: str):
    """Append a Q&A record to qa.json (best-effort)."""
    path = _qa_path()
    qa_list = _load_qa(path)
    qa_list.append({
        "when": datetime.now(timezone.utc).isoformat(),
        "session_id": st.session_state.session_id,
        "question": question,
        "answer": answer
    })
    _save_qa(path, qa_list)
# ------------------------------------------------------------------------------

# -------------------- UI --------------------
# Stylish header (nice in popup)
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
        <div style="color:#9aa3b2;font-size:13px">Groq Llama • Answers about my projects & training</div>
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
            # 0) Sarcastic-but-respectful deflection for “10 years / future vision”
            if is_future_projection_query(q):
                ans = sarcastic_future_reply()
            else:
                # 1) Guardrail for sensitive HR/negotiation topics
                deflection = hr_guardrail(q)
                if deflection:
                    ans = deflection
                else:
                    # 2) If asking for present/current projects, limit to titles with "Present"
                    if is_present_projects_query(q):
                        pres = present_projects_only(projects)
                        prompt = build_present_prompt(q, pres)
                        ans = llm_call(prompt)
                    else:
                        # 3) RAG context + LLM (normal path)
                        ctxs = retrieve(q, EMB, DOCS, k=MAX_CTX_DOCS)
                        prompt = build_prompt(q, ctxs)
                        ans = llm_call(prompt)

            st.markdown(ans)
            st.session_state.history.append({"role": "assistant", "content": ans})

            # Save to qa.json (best-effort)
            log_qa(q, ans)
