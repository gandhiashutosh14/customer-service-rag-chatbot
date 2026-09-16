"""
Deterministic parts of the chatbot, kept free of Streamlit and the LLM so they
can be unit-tested: input validators, complaint-ID parsing, prompt builders,
record formatting, and the slot-filling state machine for filing a complaint.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Tuple

INTENTS = {"file_complaint", "retrieve_complaint", "general_query"}

FIELDS = ["name", "phone_number", "email", "complaint_details"]

PROMPTS = {
    "name": "I'm sorry to hear about {issue}. Please provide your name.",
    "phone_number": "Thank you, {name}. What is your phone number?",
    "email": "Got it. Please provide your email address.",
    "complaint_details": "Thanks. Can you share more details about {issue}?",
}

ERRORS = {
    "name": "The name you provided isn't valid. Please enter your full name (letters and spaces only).",
    "phone_number": "Invalid phone number: '{value}'. Please enter a valid 10-15 digit phone (optional '+' prefix).",
    "email": "Invalid email address: '{value}'. Please enter a valid address (example: user@example.com).",
}

# Words that suggest the user typed a complaint where a name was expected.
COMPLAINT_KEYWORDS = ["complaint", "order", "delivery", "issue", "late", "wrong"]

_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------
def valid_name(value: str) -> bool:
    value = value.strip()
    if not re.fullmatch(r"[A-Za-z ]{3,50}", value):
        return False
    lowered = value.lower()
    return not any(kw in lowered for kw in COMPLAINT_KEYWORDS)


def valid_phone(value: str) -> bool:
    return bool(re.fullmatch(r"\+?\d{10,15}", value.strip()))


def valid_email(value: str) -> bool:
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value.strip()))


VALIDATORS = {"name": valid_name, "phone_number": valid_phone, "email": valid_email}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def parse_complaint_id(text: str) -> Optional[str]:
    """Return the first UUID found in free text, lower-cased, or None."""
    match = _UUID_RE.search(text)
    return match.group(0).lower() if match else None


def normalise_intent(raw: str) -> str:
    """Reduce an LLM reply to one of the known intent labels."""
    label = (raw or "").strip().splitlines()[0].strip().strip(".").lower() if raw and raw.strip() else ""
    return label if label in INTENTS else "general_query"


def format_ts(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%B %d, %Y at %I:%M %p")
    except (TypeError, ValueError):
        return ts


def format_complaint(record: Dict[str, str]) -> str:
    return (
        f"**Complaint ID**: {record['complaint_id']}  \n"
        f"**Name**: {record['name']}  \n"
        f"**Phone**: {record['phone_number']}  \n"
        f"**Email**: {record['email']}  \n"
        f"**Details**: {record['complaint_details']}  \n"
        f"**Created At**: {format_ts(record['created_at'])}"
    )


def build_intent_prompt(user_text: str) -> str:
    return f"""
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


def build_issue_brief_prompt(user_text: str) -> str:
    return f"""
Extract topic (max 3 words) of this complaint:
"{user_text}"
"""


# ---------------------------------------------------------------------------
# Slot filling
# ---------------------------------------------------------------------------
@dataclass
class SlotFiller:
    """Collects the four complaint fields one at a time, validating each.

    Usage: `prompt = SlotFiller(issue).start()`, then for each user reply
    `accepted, reply = filler.submit(text)`. When `filler.complete` is True,
    `filler.data` holds the payload for POST /complaints.
    """
    issue: str = "your issue"
    data: Dict[str, str] = field(default_factory=dict)
    index: int = 0

    @property
    def current_field(self) -> Optional[str]:
        return FIELDS[self.index] if self.index < len(FIELDS) else None

    @property
    def complete(self) -> bool:
        return self.index >= len(FIELDS)

    def start(self) -> str:
        return PROMPTS["name"].format(issue=self.issue)

    def current_prompt(self) -> str:
        """The question for the field currently being collected ("" when complete)."""
        current = self.current_field
        if current is None:
            return ""
        return PROMPTS[current].format(name=self.data.get("name", ""), issue=self.issue)

    def submit(self, text: str) -> Tuple[bool, str]:
        """Validate `text` for the current field. Returns (accepted, message).

        On acceptance the message is the prompt for the next field, or "" once
        every field is collected. On rejection it is the error to show.
        """
        current = self.current_field
        if current is None:
            return False, ""
        value = text.strip()
        validator = VALIDATORS.get(current)
        if validator and not validator(value):
            return False, ERRORS[current].format(value=value)

        self.data[current] = value
        self.index += 1
        nxt = self.current_field
        if nxt is None:
            return True, ""
        return True, PROMPTS[nxt].format(name=self.data.get("name", ""), issue=self.issue)
