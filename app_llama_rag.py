# app_openai_rag.py
# Streamlit RAG over assets/projects.json and assets/further_training.json
# Uses Sentence-Transformers + NumPy cosine retrieval (no FAISS) + OpenAI Chat Completions
# HR-aware: deflects sensitive/negotiation questions with a friendly CTA.
# Voice: warm, confident, persuasive, FIRST PERSON (I / me / my).
# Now with robust per-question JSONL logging (+ daily rotation) and a stronger prompt.

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

from datetime import datetime, timezone
import uuid

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

# -------------------- OpenAI Config --------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

MAX_CTX_DOCS = 6
TEMPERATURE = 0.25
MAX_TOKENS = 550
MAX_ANSWER_WORDS = int(os.getenv("MAX_ANSWER_WORDS", "180"))

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
            f"Please call me at **{MY_PHONE}**.\n\n"
            f"If it helps, drop me an email at **{MY_EMAIL}** with a couple of time slots."
        )
    return None

# -------------------- Fun/Sarcasm Guardrail --------------------
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
    lines = [
        "Ah, the crystal-ball classic. In ten years I’ll be exactly ten years older—that forecast has a 100% hit rate so far.",
        "I keep my horizon practical: learn fast, ship value, and iterate.",
        "If you want proof, ask me about shipped results; I’m happy to dive in.",
    ]
    return f"{lines[0]}\n\n- {lines[1]}\n- {lines[2]}"

# -------------------- Logging (JSON + JSONL with daily rotation) --------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

def _logs_dir() -> Path:
    for base in LOCAL_CANDIDATES:
        try:
            (base / "logs").mkdir(parents=True, exist_ok=True)
            return base / "logs"
        except Exception:
            continue
    p = Path("./logs")
    p.mkdir(parents=True, exist_ok=True)
    return p

def _qa_json_path() -> Path:
    # Persistent rolling JSON list (not rotated)
    logs = _logs_dir()
    return logs / "qa.json"

def _questions_jsonl_path() -> Path:
    # Rotated daily JSONL
    logs = _logs_dir()
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return logs / f"{day}-questions.jsonl"

def _append_jsonl(path: Path, rec: Dict[str, Any]):
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

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
        pass

def log_question(question: str):
    # Lightweight append-only line per question
    qrec = {
        "when": datetime.now(timezone.utc).isoformat(),
        "session_id": st.session_state.session_id,
        "question": question,
        "user_agent": st.session_state.get("_browser", None),  # may be None; Streamlit doesn't expose UA officially
        "model": OPENAI_MODEL,
        "source": "streamlit_chat_input"
    }
    _append_jsonl(_questions_jsonl_path(), qrec)

def log_qa(question: str, answer: str):
    # Also keep a compact JSON array with Q&A pairs
    path = _qa_json_path()
    qa_list = _load_qa(path)
    qa_list.append({
        "when": datetime.now(timezone.utc).isoformat(),
        "session_id": st.session_state.session_id,
        "question": question,
        "answer": answer
    })
    _save_qa(path, qa_list)

# -------------------- LLM (OpenAI) --------------------
def _system_prompt() -> str:
    """Sophisticated, HR-aware, RAG-constrained system prompt."""
    return textwrap.dedent(f"""
    <role>
    You are Kened Kqiraj’s personal portfolio assistant.
    You always write in FIRST PERSON as Kened (“I”, “me”, “my”), in warm, confident, concise prose.
    You can hold a multi-turn conversation, but you must always answer ONLY the latest user question.
    Never bring up previous answers or actions unless the user explicitly asks you to.
    Primary goal: help a hiring manager or recruiter quickly understand why I’m a strong fit, using ONLY the provided context.
    </role>

    <scope>
    You are ONLY allowed to answer questions about:
    - My projects, experience, responsibilities, results, stack/tools, and ways of working.
    - My education, courses, further training, and relevant skills.
    - How I approach problems, collaboration, or roles, as long as it is grounded in the context.
    If the user asks anything unrelated (for example: health, politics, news, random facts, private life, or topics not supported by context),
    you MUST answer with a short apology and say that you are only allowed to answer questions about my projects, skills, and training.
    Do NOT try to improvise or answer off-topic questions.
    </scope>

    <profile>
    PROFILE FACTS: {PROFILE}
    Contact: Phone {MY_PHONE} • Email {MY_EMAIL}
    </profile>

    <policy>
    - Use ONLY the RAG context chunks and the profile facts. If a detail isn't in context, say I don't have that detail here.
    - Do NOT invent projects, companies, dates, metrics, or technologies.
    - Prefer short paragraphs and tight bullet points. Avoid fluff and buzzwords.
    - Highlight outcomes, impact, metrics, and responsibilities.
    - If asked about salary/compensation/benefits/visa/contract terms/etc., do NOT provide specifics.
      Politely redirect to a call or email using the contact info above.
    - If asked purely speculative “future projection” questions, it’s ok to use one friendly, light line of humor,
      then pivot to evidence of delivered results.
    - Keep most answers within ~{MAX_ANSWER_WORDS} words unless the user explicitly asks for more depth.
    - Never describe or explain what you did internally (retrieval, logging, models, JSON, etc.).
      The user only sees your final answer.
    </policy>

    <formatting-preferences>
    - When listing items, use bullet points with actionable verbs and specific results.
    - If asked to compare or outline steps, use numbered lists.
    - Use brief **bold** callouts sparingly to emphasize outcomes or tools.
    </formatting-preferences>

    <answer-framework>
    For on-topic questions:
      1) Start with a one-sentence hook (impact-oriented and relevant to the question).
      2) Then provide 3–6 bullets covering: problem/context → what I built/did → stack/tools → measurable outcome.
      3) End with a short CTA: offer to share a repo, demo, or deeper details.
    For off-topic questions:
      - Say something like: "Sorry, I’m not allowed to answer that. I can only answer questions about my projects, skills, and training."
    Always focus on answering exactly what the user asked, nothing more.
    </answer-framework>
    """)

def llm_call(prompt: str) -> str:
    if not OPENAI_API_KEY:
        return ("Hosted LLM missing: set OPENAI_API_KEY.\n"
                "Streamlit Cloud: App → Settings → Secrets → add OPENAI_API_KEY.\n"
                "Local: export OPENAI_API_KEY=...")

    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": _system_prompt()},
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
    """User-visible prompt given to the LLM (RAG-constrained)."""
    system = textwrap.dedent(f"""
    You are answering ONE specific question from a recruiter/hiring manager about my projects, skills, experience, or training.
    Use only the context bullets below (plus the short profile facts) to answer.
    If the question is unrelated to my projects/experience/skills/training, you MUST politely refuse and say you are only allowed
    to answer questions about those topics.
    If the answer isn't in the context, say that I don't have that detail here rather than inventing it.
    Always answer ONLY the latest question; do not recap past questions unless explicitly asked.
    Write as Kened in the first person (I / me / my).
    Keep it focused, persuasive, and hiring-oriented.
    Aim for <= {MAX_ANSWER_WORDS} words unless asked otherwise.
    """).strip()

    ctx = "\n".join(f"- {c}" for c in contexts) if contexts else "- (no context available)"
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
    system = textwrap.dedent(f"""
    You are Kened’s assistant describing ONLY the projects whose title contains the word "Present".
    Make it engaging and concise for a hiring audience:
    - Start with a one-sentence hook (impact/outcome).
    - Then give 3–5 punchy bullets: what I'm building, stack, why it matters, concrete results.
    - Keep it first-person, confident, concrete.
    - If nothing is available, say I don't have a project marked as present here.
    Limit to ~{MAX_ANSWER_WORDS} words unless the user asks for depth.
    """).strip()
    ctx = "\n".join(f"- {c}" for c in contexts) if contexts else "- (no present projects found)"
    user = f"Question: {question}\n\nContext (only 'Present' projects):\n{ctx}\n\nAnswer:"
    return f"{system}\n\n{user}"
# -----------------------------------------------------------------------------

# -------------------- UI --------------------
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
        <div style="color:#9aa3b2;font-size:13px">OpenAI • Answers about my projects & training</div>
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

# --- 🔥 WARMUP ON SERVER START (once per process) -----------------------------
# Ensures the embedder, retrieval, and OpenAI call are hot to avoid cold start.
if "server_warmed" not in st.session_state:
    try:
        _ = get_embedder()                     # load SentenceTransformer into memory
        _ = retrieve("hello", EMB, DOCS, k=2)  # force an embedding/query pass
        _ = llm_call("Say a short hello to confirm warmup.")  # ping OpenAI backend
    except Exception:
        pass
    st.session_state.server_warmed = True
# ------------------------------------------------------------------------------

# Warm first message
if "history" not in st.session_state:
    st.session_state.history = [
        {"role": "assistant",
         "content": "Hi! I’m Kened’s assistant. Ask me about my projects, stack, experience, or training."}
    ]

# History render
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Small sidebar tools
with st.sidebar:
    st.subheader("⚙️ Controls")
    st.caption("Answer length & model")
    _max_words = st.slider("Max answer words", 80, 400, MAX_ANSWER_WORDS, step=10)
    MAX_ANSWER_WORDS = _max_words  # update live for this session

    st.text_input("OpenAI model", value=OPENAI_MODEL, key="model_name_sidebar")
    if st.button("Apply model"):
        OPENAI_MODEL = st.session_state.get("model_name_sidebar", OPENAI_MODEL)
        st.success(f"Model set to: {OPENAI_MODEL}")

    st.divider()
    st.subheader("🗂 Logs")
    qa_path = _qa_json_path()
    ql_path = _questions_jsonl_path()
    st.caption(f"Q&A JSON: {qa_path}")
    st.caption(f"Questions JSONL (daily): {ql_path}")
    if qa_path.exists():
        st.download_button("Download qa.json", qa_path.read_bytes(), file_name="qa.json")
    if ql_path.exists():
        st.download_button("Download questions.jsonl", ql_path.read_bytes(), file_name=ql_path.name)

# Chat input
q = st.chat_input("Type your question…")

if q:
    # 1) record the question immediately (append-only JSONL)
    log_question(q)

    # 2) show user message
    st.session_state.history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    # 3) answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            if is_future_projection_query(q):
                ans = sarcastic_future_reply()
            else:
                deflection = hr_guardrail(q)
                if deflection:
                    ans = deflection
                else:
                    if is_present_projects_query(q):
                        pres = present_projects_only(projects)
                        prompt = build_present_prompt(q, pres)
                        ans = llm_call(prompt)
                    else:
                        ctxs = retrieve(q, EMB, DOCS, k=MAX_CTX_DOCS)
                        prompt = build_prompt(q, ctxs)
                        ans = llm_call(prompt)

            # 4) show + history + log Q&A
            st.markdown(ans)
            st.session_state.history.append({"role": "assistant", "content": ans})
            log_qa(q, ans)
