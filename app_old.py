import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import os
import requests
from dotenv import load_dotenv
import re
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
API_URL = "http://127.0.0.1:8000"  # Consistent with Uvicorn

# Initialize LLM
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")

# Initialize RAG components
def initialize_rag():
    documents = []
    for filename in os.listdir("knowledge_base"):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join("knowledge_base", filename))
            documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = [
        {"role": "assistant", "content": "Hello! How can I assist you today? You can ask questions or file a complaint."}
    ]
if "complaint_mode" not in st.session_state:
    st.session_state.complaint_mode = False
if "complaint_data" not in st.session_state:
    st.session_state.complaint_data = {}
if "last_complaint_id" not in st.session_state:
    st.session_state.last_complaint_id = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = initialize_rag()

# Function to validate phone number
def validate_phone_number(phone):
    if re.match(r"^\+?\d{10,15}$", phone):
        return phone if phone.startswith("+") else "+" + phone
    raise ValueError("Phone number must be 10-15 digits, optionally starting with +")

# Function to classify intent
def classify_intent(query):
    prompt = f"""
    Classify the user's query into one of the following intents:
    - file_complaint: The user wants to submit a new complaint.
    - retrieve_complaint: The user wants to get details about an existing complaint.
    - general_query: The user has a general question.

    Examples:
    User: I want to file a complaint about a delayed delivery.
    Intent: file_complaint

    User: Show details for complaint XYZ123.
    Intent: retrieve_complaint

    User: What are your business hours?
    Intent: general_query

    User: Can you show me the details of my complaint #12345?
    Intent: retrieve_complaint

    User: What's the status of my order using my complaint ID ABCDEF?
    Intent: retrieve_complaint

    User: My order hasn't arrived yet.
    Intent: file_complaint

    User: How do I file a complaint?
    Intent: general_query

    Query: {query}
    Intent:
    """
    response = llm.invoke(prompt)
    intent = response.content.strip().lower()
    return intent

# Function to extract complaint details
def extract_complaint_details(user_input):
    prompt = f"""
    Extract the following details from the user's message:
    - name
    - phone_number
    - email
    - complaint_details

    If a detail is not provided, leave it blank.

    User: {user_input}
    Details:
    """
    response = llm.invoke(prompt)
    details = response.content.strip().split('\n')
    extracted_data = {}
    for detail in details:
        if ':' in detail:
            key, value = detail.split(':', 1)
            key = key.strip().lower().replace(' ', '_')
            value = value.strip()
            if value:
                extracted_data[key] = value
    # Validate phone number
    if 'phone_number' in extracted_data:
        try:
            extracted_data['phone_number'] = validate_phone_number(extracted_data['phone_number'])
        except ValueError as e:
            extracted_data['phone_number_error'] = str(e)
    return extracted_data

# Function to extract complaint ID
def extract_complaint_id(prompt):
    cleaned_prompt = re.sub(r'[^a-zA-Z0-9-]', '', prompt)
    uuid_pattern = r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}'
    match = re.search(uuid_pattern, cleaned_prompt, re.IGNORECASE)
    if match:
        return match.group(0)
    return None

# Display chat history
st.title("Customer Service Chatbot")
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("What is your question or complaint?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.history.append({"role": "user", "content": prompt})

    # Process input
    if not st.session_state.complaint_mode:
        intent = classify_intent(prompt)
        if intent == "file_complaint":
            st.session_state.complaint_mode = True
            st.session_state.complaint_data = {}
            response = "I'm sorry to hear about your issue. Please provide your name, phone number (10-15 digits, e.g., +1234567890), email, and a brief description of your problem."
        elif intent == "retrieve_complaint":
            complaint_id = extract_complaint_id(prompt)
            if not complaint_id and st.session_state.last_complaint_id:
                complaint_id = st.session_state.last_complaint_id
            if complaint_id:
                try:
                    logger.debug(f"Sending GET request to {API_URL}/complaints/{complaint_id}")
                    api_response = requests.get(f"{API_URL}/complaints/{complaint_id}", timeout=5)
                    api_response.raise_for_status()
                    logger.debug(f"GET /complaints/{complaint_id} - Status: {api_response.status_code}, Response: {api_response.text}")
                    details = api_response.json()
                    response = f"""
                    **Complaint ID**: {details['complaint_id']}
                    **Name**: {details['name']}
                    **Phone**: {details['phone_number']}
                    **Email**: {details['email']}
                    **Details**: {details['complaint_details']}
                    **Created At**: {details['created_at']}
                    """
                except requests.exceptions.RequestException as e:
                    logger.error(f"Error retrieving complaint {complaint_id}: {str(e)}")
                    response = f"Error retrieving complaint: {str(e)}"
            else:
                response = "Please provide a valid complaint ID (e.g., 123e4567-e89b-12d3-a456-426614174000)."
        else:
            rag_response = st.session_state.rag_chain.invoke({"query": prompt})
            response = rag_response["result"]
    else:
        if "cancel" in prompt.lower():
            st.session_state.complaint_mode = False
            st.session_state.complaint_data = {}
            response = "Complaint filing canceled."
        else:
            extracted_data = extract_complaint_details(prompt)
            st.session_state.complaint_data.update(extracted_data)
            logger.debug(f"Complaint data after update: {st.session_state.complaint_data}")
            if 'phone_number_error' in st.session_state.complaint_data:
                response = f"Error: {st.session_state.complaint_data['phone_number_error']}. Please provide a valid phone number (10-15 digits, e.g., +1234567890)."
                del st.session_state.complaint_data['phone_number_error']
            else:
                missing_fields = [field for field in ["name", "phone_number", "email", "complaint_details"] if field not in st.session_state.complaint_data]
                logger.debug(f"Missing fields: {missing_fields}")
                if missing_fields:
                    next_field = missing_fields[0]
                    response = f"Please provide your {next_field.replace('_', ' ')}."
                else:
                    logger.debug(f"Preparing to send POST request to {API_URL}/complaints")
                    try:
                        complaint_data = {
                            "name": st.session_state.complaint_data["name"],
                            "phone_number": st.session_state.complaint_data["phone_number"],
                            "email": st.session_state.complaint_data["email"],
                            "complaint_details": st.session_state.complaint_data["complaint_details"]
                        }
                        logger.debug(f"POST request data: {complaint_data}")
                        api_response = requests.post(f"{API_URL}/complaints", json=complaint_data, timeout=5)
                
                        api_response.raise_for_status()
                        logger.debug(f"POST /complaints - Status: {api_response.status_code}, Response: {api_response.text}")
                        complaint_id = api_response.json()["complaint_id"]
                        st.session_state.last_complaint_id = complaint_id
                        response = f"Your complaint has been registered with ID: {complaint_id}. We'll look into this and get back to you soon."
                        st.session_state.complaint_mode = False
                        st.session_state.complaint_data = {}
                    except requests.exceptions.HTTPError as e:
                        logger.error(f"HTTP error creating complaint: {str(e)}, Response: {e.response.text if e.response else 'No response'}")
                        response = f"Error creating complaint: {str(e)}. Please try again."
                    except requests.exceptions.ConnectionError as e:
                        logger.error(f"Connection error creating complaint: {str(e)}")
                        response = "Error: Could not connect to the server. Please check if the server is running and try again."
                    except requests.exceptions.Timeout as e:
                        logger.error(f"Timeout error creating complaint: {str(e)}")
                        response = "Error: Server request timed out. Please try again later."
                    except requests.exceptions.RequestException as e:
                        logger.error(f"General request error creating complaint: {str(e)}")
                        response = f"Error creating complaint: {str(e)}. Please try again."

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.history.append({"role": "assistant", "content": response})