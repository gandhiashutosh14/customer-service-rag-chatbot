"""
Streamlit chat front end: RAG answers over the knowledge base, plus a
slot-filling flow that files complaints through the FastAPI service and a
retrieval flow that fetches them by ID. Deterministic logic lives in
chat_logic.py; this file owns the UI and the LLM calls.
"""
import logging
import os

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from chat_logic import (
    SlotFiller,
    build_intent_prompt,
    build_issue_brief_prompt,
    format_complaint,
    normalise_intent,
    parse_complaint_id,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Configuration ─────────────────────────────────────────
load_dotenv()
API_URL = os.getenv("COMPLAINTS_API_URL", "http://127.0.0.1:8000")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
KNOWLEDGE_DIR = os.getenv("KNOWLEDGE_DIR", "knowledge_base")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

st.set_page_config(page_title="Customer Service Chatbot", page_icon="💬")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    st.stop()

llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL)


# ─── RAG chain (built once per process) ────────────────────
@st.cache_resource(show_spinner="Indexing knowledge base...")
def build_rag_chain(knowledge_dir: str, embedding_model: str):
    docs = []
    for fname in sorted(os.listdir(knowledge_dir)):
        if fname.lower().endswith(".pdf"):
            docs.extend(PyPDFLoader(os.path.join(knowledge_dir, fname)).load())
    if not docs:
        raise RuntimeError(f"No PDF files found in {knowledge_dir}/")
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    embedder = HuggingFaceEmbeddings(model_name=embedding_model)
    retriever = FAISS.from_documents(chunks, embedder).as_retriever(search_kwargs={"k": 5})
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


# ─── LLM helpers ───────────────────────────────────────────
def classify_intent(user_text: str) -> str:
    try:
        return normalise_intent(llm.invoke(build_intent_prompt(user_text)).content)
    except Exception as exc:  # noqa: BLE001 - degrade to RAG rather than crash the chat
        logger.error("Intent classification failed: %s", exc)
        return "general_query"


def extract_issue_brief(user_text: str) -> str:
    try:
        return llm.invoke(build_issue_brief_prompt(user_text)).content.strip() or "your issue"
    except Exception:  # noqa: BLE001
        return "your issue"


# ─── Complaint API helpers ─────────────────────────────────
def create_complaint(payload: dict) -> str:
    r = requests.post(f"{API_URL}/complaints", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()["complaint_id"]


def fetch_complaint(complaint_id: str) -> str:
    r = requests.get(f"{API_URL}/complaints/{complaint_id}", timeout=10)
    if r.status_code == 404:
        return f"No complaint found with ID {complaint_id}."
    r.raise_for_status()
    return format_complaint(r.json())


# ─── Session state ─────────────────────────────────────────
ss = st.session_state
ss.setdefault("history", [{
    "role": "assistant",
    "content": "Hello! How can I assist you today? You can ask questions or file a complaint.",
}])
ss.setdefault("filler", None)          # SlotFiller while a complaint is being collected
ss.setdefault("pending_retrieve", None)  # complaint ID awaiting yes/no while a complaint is in progress
ss.setdefault("last_complaint_id", None)

try:
    rag_chain = build_rag_chain(KNOWLEDGE_DIR, EMBEDDING_MODEL)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not build the knowledge base: {exc}")
    st.stop()


def say(text: str) -> None:
    ss.history.append({"role": "assistant", "content": text})
    with st.chat_message("assistant"):
        st.markdown(text)


# ─── UI ────────────────────────────────────────────────────
st.title("Customer Service Chatbot")
for msg in ss.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if ss.filler is not None and st.button("Cancel Complaint"):
    ss.filler = None
    ss.pending_retrieve = None
    say("Complaint process canceled.")

if user_input := st.chat_input("Your message..."):
    usr = user_input.strip()
    ss.history.append({"role": "user", "content": usr})
    with st.chat_message("user"):
        st.markdown(usr)

    # 1. Pending yes/no: abandon the in-progress complaint to retrieve another?
    if ss.pending_retrieve:
        cid = ss.pending_retrieve
        ss.pending_retrieve = None
        if usr.lower() in ("yes", "y"):
            ss.filler = None
            try:
                say(fetch_complaint(cid))
            except Exception as exc:  # noqa: BLE001
                say(f"Error retrieving complaint: {exc}")
        else:
            say(f"Okay, continuing your complaint. {ss.filler.current_prompt()}")
        st.stop()

    # 2. Slot filling takes priority while a complaint is in progress
    if ss.filler is not None:
        cid = parse_complaint_id(usr)
        if cid and ss.filler.current_field != "complaint_details":
            ss.pending_retrieve = cid
            say("You have an in-progress complaint. Cancel it and retrieve the other one? (yes/no)")
            st.stop()

        accepted, reply = ss.filler.submit(usr)
        if not accepted:
            say(reply)
        elif ss.filler.complete:
            try:
                complaint_id = create_complaint(ss.filler.data)
                ss.last_complaint_id = complaint_id
                say(f"Your complaint has been registered with ID: {complaint_id}. We'll get back to you soon.")
            except Exception as exc:  # noqa: BLE001
                say(f"Error creating complaint: {exc}")
            ss.filler = None
        else:
            say(reply)
        st.stop()

    # 3. Fresh message: classify intent
    intent = classify_intent(usr)
    if intent == "file_complaint":
        ss.filler = SlotFiller(issue=extract_issue_brief(usr))
        say(ss.filler.start())

    elif intent == "retrieve_complaint":
        cid = parse_complaint_id(usr) or ss.last_complaint_id
        if cid:
            try:
                say(fetch_complaint(cid))
            except Exception as exc:  # noqa: BLE001
                say(f"Error retrieving complaint: {exc}")
        else:
            say("Please provide a valid complaint ID.")

    else:
        try:
            result = rag_chain.invoke({"query": usr})
            say(result.get("result") or "Sorry, I couldn't find an answer.")
        except Exception as exc:  # noqa: BLE001
            say(f"RAG error: {exc}")
