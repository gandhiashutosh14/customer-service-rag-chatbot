import streamlit as st
import os
import requests
import re
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from datetime import datetime

# ─── Logging setup ─────────────────────────────────────────
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ─── Environment & API URLs ─────────────────────────────────
load_dotenv()
API_URL      = "http://127.0.0.1:8000"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ─── LLM & Embeddings ──────────────────────────────────────
llm      = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ─── Build RAG Retriever ───────────────────────────────────
def init_rag():
    docs = []
    for fname in os.listdir("knowledge_base"):
        if fname.endswith(".pdf"):
            docs.extend(PyPDFLoader(os.path.join("knowledge_base", fname)).load())
    splitter   = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks     = splitter.split_documents(docs)
    db         = FAISS.from_documents(chunks, embedder)
    retriever  = db.as_retriever()
    retriever.search_kwargs = {"k": 5}
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# ─── Few-shot intent classification ─────────────────────────
def classify_intent(user_text: str) -> str:
    prompt = f"""
You are an intent classifier. Classify into file_complaint, retrieve_complaint, or general_query.
Respond exactly with the label.

Examples:
User: I want to file a complaint about a delayed delivery.
Intent: file_complaint

User: Show details for complaint ABC123.
Intent: retrieve_complaint

User: mujhe ek complaint hai galat order ke liye
Intent: file_complaint

User: What time do you close?
Intent: general_query

Now classify:
User: {user_text}
Intent:
"""
    try:
        res = llm.invoke(prompt).content.strip().splitlines()[0]
        return res if res in {"file_complaint","retrieve_complaint","general_query"} else "general_query"
    except Exception as e:
        logger.error("Intent classification failed: %s", e)
        return "general_query"

# ─── Extract brief complaint topic ─────────────────────────
def extract_issue_brief(user_text: str) -> str:
    prompt = f"""
Extract topic (max 3 words) of this complaint:
"{user_text}"
"""
    try:
        return llm.invoke(prompt).content.strip() or "your issue"
    except:
        return "your issue"

# ─── Session state init ────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = [{
        "role":"assistant",
        "content":"Hello! How can I assist you today? You can ask questions or file a complaint."
    }]
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = init_rag()
for var in ("complaint_mode","complaint_data","current_field","pending_action","last_complaint_id","issue_brief"):
    if var not in st.session_state:
        st.session_state[var] = False if var=="complaint_mode" else {} if var=="complaint_data" else None

# ─── Complaint flow config ────────────────────────────────
FIELDS = ["name","phone_number","email","complaint_details"]
PROMPTS = {
    "name":              "I'm sorry to hear about {issue}. Please provide your name.",
    "phone_number":      "Thank you, {name}. What is your phone number?",
    "email":             "Got it. Please provide your email address.",
    "complaint_details": "Thanks. Can you share more details about {issue}?"
}

# ─── Validators + keywords ─────────────────────────────────
def valid_name(n):  return bool(re.fullmatch(r"[A-Za-z ]{3,50}", n))
def valid_phone(p): return bool(re.fullmatch(r"\+?\d{10,15}", p))
def valid_email(e): return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", e))
COMPLAINT_KEYWORDS = ["complaint","order","delivery","issue","late","wrong"]

# ─── Timestamp formatter ───────────────────────────────────
def format_ts(ts):
    try:
        return datetime.fromisoformat(ts).strftime("%B %d, %Y at %I:%M %p")
    except:
        return ts

# ─── UI header & history ───────────────────────────────────
st.title("Customer Service Chatbot")
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── Cancel button shows immediately once complaint_mode is True ───
if st.session_state.complaint_mode:
    if st.button("Cancel Complaint"):
        st.session_state.complaint_mode = False
        st.session_state.complaint_data.clear()
        st.session_state.current_field = None
        st.session_state.pending_action = None
        msg = "Complaint process canceled."
        st.session_state.history.append({"role":"assistant","content":msg})
        with st.chat_message("assistant"):
            st.markdown(msg)

# ─── Main interaction ──────────────────────────────────────
if user_input := st.chat_input("Your message..."):
    usr = user_input.strip()
    st.session_state.history.append({"role":"user","content":usr})
    with st.chat_message("user"):
        st.markdown(usr)

    # 1️⃣ Slot-filling priority
    if (st.session_state.complaint_mode and
        st.session_state.current_field is not None and
        not st.session_state.pending_action):

        fld   = FIELDS[st.session_state.current_field]
        name  = st.session_state.complaint_data.get("name","")
        issue = st.session_state.issue_brief or "your issue"

        if fld == "name":
            if not valid_name(usr) or any(kw in usr.lower() for kw in COMPLAINT_KEYWORDS):
                response = "The name you provided isn’t valid. Please enter your full name (letters and spaces only)."
            else:
                st.session_state.complaint_data[fld] = usr
                st.session_state.current_field += 1
                response = PROMPTS["phone_number"].format(name=usr,issue=issue)

        elif fld == "phone_number":
            if not valid_phone(usr):
                response = f"Invalid phone number: '{usr}'. Please enter a valid 10–15 digit phone (optional '+' prefix)."
            else:
                st.session_state.complaint_data[fld] = usr
                st.session_state.current_field += 1
                response = PROMPTS["email"].format(name=name,issue=issue)

        elif fld == "email":
            if not valid_email(usr):
                response = f"Invalid email address: '{usr}'. Please enter a valid address (example: user@example.com)."
            else:
                st.session_state.complaint_data[fld] = usr
                st.session_state.current_field += 1
                response = PROMPTS["complaint_details"].format(name=name,issue=issue)

        else:
            st.session_state.complaint_data[fld] = usr
            try:
                r = requests.post(f"{API_URL}/complaints", json=st.session_state.complaint_data)
                r.raise_for_status()
                cid = r.json()["complaint_id"]
                response = f"Your complaint has been registered with ID: {cid}. We'll get back to you soon."
                st.session_state.last_complaint_id = cid
            except Exception as e:
                response = f"Error creating complaint: {e}"
            st.session_state.complaint_mode = False
            st.session_state.current_field = None
            st.session_state.complaint_data.clear()

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.history.append({"role":"assistant","content":response})
        st.stop()

    # 2️⃣ Pending confirmation (in-progress → retrieve)
    if st.session_state.pending_action:
        act = st.session_state.pending_action
        if usr.lower() in ("yes","y") and act.get("type") == "retrieve":
            cid = act["id"]
            st.session_state.pending_action = None
            try:
                r = requests.get(f"{API_URL}/complaints/{cid}")
                r.raise_for_status()
                d = r.json()
                response = (
                    f"**Complaint ID**: {d['complaint_id']}  \n"
                    f"**Name**: {d['name']}              \n"
                    f"**Phone**: {d['phone_number']}    \n"
                    f"**Email**: {d['email']}           \n"
                    f"**Details**: {d['complaint_details']}\n"
                    f"**Created At**: {format_ts(d['created_at'])}"
                )
            except Exception as e:
                response = f"Error retrieving complaint: {e}"
        else:
            st.session_state.pending_action = None
            st.session_state.complaint_mode = True
            st.session_state.current_field = 0
            response = PROMPTS["name"].format(issue=st.session_state.issue_brief)

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.history.append({"role":"assistant","content":response})
        st.stop()

    # 3️⃣ Intent detection
    intent = classify_intent(usr)
    if intent == "file_complaint":
        st.session_state.issue_brief = extract_issue_brief(usr)
        st.session_state.complaint_mode = True
        st.session_state.current_field = 0
        response = PROMPTS["name"].format(issue=st.session_state.issue_brief)

    elif intent == "retrieve_complaint":
        m = re.search(r"[a-f0-9\\-]{36}", usr)
        if m:
            cid = m.group(0)
            if st.session_state.complaint_mode:
                st.session_state.pending_action = {"type":"retrieve","id":cid}
                response = "You have an in-progress complaint. Cancel it and retrieve? (yes/no)"
            else:
                try:
                    r = requests.get(f"{API_URL}/complaints/{cid}")
                    r.raise_for_status()
                    d = r.json()
                    response = (
                        f"**Complaint ID**: {d['complaint_id']}  \n"
                        f"**Name**: {d['name']}              \n"
                        f"**Phone**: {d['phone_number']}    \n"
                        f"**Email**: {d['email']}           \n"
                        f"**Details**: {d['complaint_details']}\n"
                        f"**Created At**: {format_ts(d['created_at'])}"
                    )
                except Exception as e:
                    response = f"Error retrieving complaint: {e}"
        else:
            response = "Please provide a valid complaint ID."

    else:
        # RAG fallback
        try:
            rag = st.session_state.rag_chain.invoke({"query":usr})
            response = rag.get("result","Sorry, I couldn't find an answer.")
        except Exception as e:
            response = f"RAG error: {e}"

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.history.append({"role":"assistant","content":response})
