from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr, validator
import sqlite3
import uuid
from datetime import datetime
import re

app = FastAPI()

# Database setup
conn = sqlite3.connect("complaints.db", check_same_thread=False)
conn.execute("""
    CREATE TABLE IF NOT EXISTS complaints (
        id TEXT PRIMARY KEY,
        name TEXT,
        phone TEXT,
        email TEXT,
        complaint_details TEXT,
        created_at TEXT
    )
""")

# Pydantic model for complaint
class Complaint(BaseModel):
    name: str
    phone_number: str
    email: EmailStr
    complaint_details: str

    @validator("phone_number")
    def validate_phone(cls, v):
        if not re.match(r"^\+?\d{10,15}$", v):
            raise ValueError("Invalid phone number format; must be 10-15 digits, optional leading +")
        return v

    @validator("name")
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v

    @validator("complaint_details")
    def validate_details(cls, v):
        if not v.strip():
            raise ValueError("Complaint details cannot be empty")
        return v

# POST endpoint
@app.post("/complaints")
async def create_complaint(complaint: Complaint):
    complaint_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    try:
        conn.execute(
            "INSERT INTO complaints VALUES (?, ?, ?, ?, ?, ?)",
            (
                complaint_id,
                complaint.name,
                complaint.phone_number,
                complaint.email,
                complaint.complaint_details,
                created_at,
            ),
        )
        conn.commit()
        return {"complaint_id": complaint_id, "message": "Complaint created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# GET endpoint
@app.get("/complaints/{complaint_id}")
async def get_complaint(complaint_id: str):
    cursor = conn.execute("SELECT * FROM complaints WHERE id = ?", (complaint_id,))
    row = cursor.fetchone()
    if row:
        return {
            "complaint_id": row[0],
            "name": row[1],
            "phone_number": row[2],
            "email": row[3],
            "complaint_details": row[4],
            "created_at": row[5],
        }
    else:
        raise HTTPException(status_code=404, detail="Complaint not found")
