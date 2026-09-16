"""
Complaint service: a small FastAPI app backed by SQLite.

  POST /complaints              create a complaint, returns its UUID
  GET  /complaints/{id}         fetch a complaint
  GET  /health                  liveness probe

The database path comes from COMPLAINTS_DB_PATH (default: ./complaints.db),
which is what lets the test-suite run against a temporary file.
"""
from __future__ import annotations

import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr, field_validator

DB_PATH = os.environ.get("COMPLAINTS_DB_PATH", "complaints.db")

app = FastAPI(title="Complaint service", version="1.1.0")

_conn: Optional[sqlite3.Connection] = None


def get_conn() -> sqlite3.Connection:
    """Open the SQLite connection on first use and make sure the table exists."""
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS complaints (
                id TEXT PRIMARY KEY,
                name TEXT,
                phone TEXT,
                email TEXT,
                complaint_details TEXT,
                created_at TEXT
            )
            """
        )
        _conn.commit()
    return _conn


class Complaint(BaseModel):
    name: str
    phone_number: str
    email: EmailStr
    complaint_details: str

    @field_validator("phone_number")
    @classmethod
    def validate_phone(cls, v: str) -> str:
        if not re.fullmatch(r"\+?\d{10,15}", v.strip()):
            raise ValueError("Invalid phone number format; must be 10-15 digits, optional leading +")
        return v.strip()

    @field_validator("name", "complaint_details")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


@app.post("/complaints", status_code=201)
async def create_complaint(complaint: Complaint):
    complaint_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    conn = get_conn()
    conn.execute(
        "INSERT INTO complaints VALUES (?, ?, ?, ?, ?, ?)",
        (complaint_id, complaint.name, complaint.phone_number, complaint.email,
         complaint.complaint_details, created_at),
    )
    conn.commit()
    return {"complaint_id": complaint_id, "message": "Complaint created successfully"}


@app.get("/complaints/{complaint_id}")
async def get_complaint(complaint_id: str):
    row = get_conn().execute("SELECT * FROM complaints WHERE id = ?", (complaint_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Complaint not found")
    return {
        "complaint_id": row[0],
        "name": row[1],
        "phone_number": row[2],
        "email": row[3],
        "complaint_details": row[4],
        "created_at": row[5],
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "db": DB_PATH}
