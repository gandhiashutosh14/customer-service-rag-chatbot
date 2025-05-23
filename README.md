# Customer Service RAG Chatbot

This repository contains a Retrieval-Augmented Generation (RAG) chatbot built with Streamlit and FastAPI. The chatbot can:

* Answer general questions by retrieving context from a PDF-based knowledge base
* Collect customer complaint details (name, phone number, email, complaint description)
* Create a complaint record via a REST API and return a unique complaint ID
* Retrieve and display complaint details when given a valid complaint ID

## Features

1. Contextual responses driven by a vector store (FAISS) over PDF documents
2. Multi-step slot filling for complaint creation with immediate validation for name, phone, and email
3. FastAPI backend with two endpoints:

   * `POST /complaints` to create a new complaint
   * `GET /complaints/{complaint_id}` to fetch an existing complaint
4. Real-time retrieval of complaint records in the chatflow
5. Human-friendly timestamp formatting

## Tech Stack

* Front end: Streamlit
* Backend: FastAPI, SQLite
* Vector store: FAISS
* LLM inference: Groq Cloud (via OpenAI-compatible endpoint) or any OpenAI-compatible model
* Embeddings: sentence-transformers/all-MiniLM-L6-v2
* Language: Python 3.9+

## Prerequisites

* Python 3.9 or newer
* [poetry](https://python-poetry.org/) or `pip` for dependency management
* A Groq Cloud API key (or OpenAI API key if using the OpenAI-compatible endpoint)

## Setup

1. Clone the repository

   ```bash
   git clone https://github.com/your-username/customer-service-rag-chatbot.git
   cd customer-service-rag-chatbot
   ```

2. Install dependencies

   Using pip:

   ```bash
   pip install -r requirements.txt
   ```

   Or using poetry:

   ```bash
   poetry install
   ```

3. Create a `.env` file in the project root and set your API key:

   ```text
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. Place your sample PDF files in the `knowledge_base/` folder. These files will be loaded into FAISS.

## Running the Application

1. Start the FastAPI backend:

   ```bash
   uvicorn api:app --reload
   ```

2. In a new terminal, launch the Streamlit front end:

   ```bash
   streamlit run app.py
   ```

3. Open your browser to `http://localhost:8501` to interact with the chatbot.

## Usage

* To ask a general question, just type it in the chat box.
* To file a new complaint, start with a phrase like "I want to file a complaint" or "register an issue". The bot will prompt you for missing details one by one.
* To retrieve an existing complaint, type "show details for complaint \<complaint\_id>" or any similar phrase. The bot accepts variations like "fetch complaint", "view complaint status", etc.

## Project Structure

```
├── api.py           # FastAPI complaint service
├── app.py           # Streamlit chat interface
├── knowledge_base/  # PDF files used for RAG
├── complaints.db    # SQLite database (auto-generated)
├── requirements.txt # Python dependencies
├── README.md        # Project documentation
└── .env             # API keys (not committed)
```

## Testing & Validation

* The chat front end immediately validates each field:

  * Names must use letters and spaces only
  * Phone numbers must be 10 to 15 digits, optional leading plus sign
  * Emails must follow the standard local\@domain format
* Invalid IDs or malformed UUIDs result in a clear error message.

## Next Steps

* Add support for more languages by integrating a translation step before intent detection.
* Extend slot-filling to allow mid-flow edits or field corrections.
* Deploy on a cloud platform (Heroku, AWS, GCP) for live demo.

## License

This project is released under the MIT License. Feel free to use and modify it for your needs.
