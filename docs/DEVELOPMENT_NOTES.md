# Development notes

How the chatbot was built, refined and verified.

## Original development (May 2025)

The chatbot was written by Ashutosh Gandhi and pushed to this repository on 2025-05-23 in two
commits: the Streamlit app, the FastAPI complaint service, a sample FAQ knowledge base and a
README. The Groq API key was kept in a `.env` file that was never committed (checked with
`git log --all -- .env`).

## Refinement, 2026-09-16

### Planning

- Starting point: a clean personal project with a good README but no tests and no Docker
  packaging. The plan: work on a fresh clone of the GitHub repository, keep the local key out of
  git, add tests and a Dockerfile, and bring the README up to a recruiter-facing standard.

### What was found on inspection

- `requirements.txt` was a single space-separated line, so `pip install -r` could not parse it,
  and two packages the app imports (`langchain-huggingface`, `langchain-groq`) were missing.
- `api.py` opened `complaints.db` at import time with a hardcoded path, which made it untestable
  without writing to the real database, and used Pydantic v1-style validators.
- `app.py` mixed Streamlit rendering, LLM calls, validation and conversation state in one script,
  so none of the logic could be tested. The complaint-ID regex contained a doubled backslash that
  also matched a literal backslash.
- A stale copy, `app_old.py`, was tracked in the repository.

### Iterations

1. Cloned the repository fresh and created a virtual environment (`streamlit 1.64`, `langchain
   0.3.30`, `langchain-groq 0.3.8`, `langchain-huggingface 0.3.1`, `sentence-transformers 6.0.1`,
   `faiss-cpu 1.15`, `fastapi 0.141`, `torch 2.14`).
2. Extracted the deterministic logic into `chat_logic.py`: validators, UUID parsing, intent
   normalisation, timestamp and record formatting, prompt builders, and a `SlotFiller` state
   machine that replaces the index-and-dict bookkeeping in the original script. Behaviour was kept
   the same, including the rule that a "name" containing complaint vocabulary is rejected.
3. Rewrote `app.py` on top of it, with the RAG chain cached per process, configuration from
   environment variables, and a clear error when no key is set.
4. Rewrote `api.py` with a lazily opened connection, `COMPLAINTS_DB_PATH`, Pydantic v2
   `field_validator`s, `201` on create, and a `/health` route. Request and response shapes the UI
   depends on were left unchanged.
5. Fixed `requirements.txt`, added `requirements-dev.txt`, `.env.example`, `scripts/start.sh`, a
   `Dockerfile`, a GitHub Actions workflow, and removed `app_old.py`.
6. Wrote 21 tests: 13 for `chat_logic` (validators, parsing, normalisation, the full slot-filling
   path, rejection and re-ask, post-completion behaviour) and 8 for the API against a temporary
   SQLite file.

### Debugging

- The first draft of the "pending retrieve, user said no" branch in `app.py` contained a leftover
  reference to an undefined helper guarded by `if False`. Caught on re-read before any test ran;
  fixed by adding `SlotFiller.current_prompt()` and a test for it.
- No test failures occurred once the code was written; the suite passed on the first run.

### Verification

| Check | Result |
|---|---|
| `pytest -q` | 21 passed in 0.38s |
| `python -c "import chat_logic, api"` | OK |
| `uvicorn api:app` + curl create / fetch / unknown / invalid | 201, 200, 404, 422 |
| Streamlit UI rendered locally with a placeholder key | greeting and input rendered (screenshot in README) |

**Not verified:** a live Groq conversation and the RAG answer path (no API key was used), and the
Docker image (no Docker on the development machine). The README says so.

### Outcome

Pushed to the existing public repository https://github.com/gandhiashutosh14/customer-service-rag-chatbot
as three commits on top of the original two: logic extraction and API hardening, tests and CI, then
documentation and packaging.
