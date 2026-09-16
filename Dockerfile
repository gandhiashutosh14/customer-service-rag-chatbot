# Runs both services (API on 8000, Streamlit on 8501) in one container.
# Not built yet (no Docker on the development machine); see docs/DEVELOPMENT_NOTES.md.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x scripts/start.sh

EXPOSE 8000 8501
CMD ["scripts/start.sh"]
