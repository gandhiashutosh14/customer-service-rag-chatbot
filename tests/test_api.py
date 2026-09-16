import importlib
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Import api.py against a temporary SQLite file so tests never touch complaints.db."""
    monkeypatch.setenv("COMPLAINTS_DB_PATH", str(tmp_path / "test.db"))
    sys.modules.pop("api", None)
    api = importlib.import_module("api")
    with TestClient(api.app) as c:
        yield c


VALID = {
    "name": "Priya Sharma",
    "phone_number": "+919876543210",
    "email": "priya@example.com",
    "complaint_details": "Parcel arrived damaged.",
}


def test_create_then_fetch_round_trip(client):
    created = client.post("/complaints", json=VALID)
    assert created.status_code == 201
    complaint_id = created.json()["complaint_id"]
    assert len(complaint_id) == 36

    fetched = client.get(f"/complaints/{complaint_id}")
    assert fetched.status_code == 200
    body = fetched.json()
    assert body["complaint_id"] == complaint_id
    assert body["name"] == VALID["name"]
    assert body["email"] == VALID["email"]
    assert body["created_at"].endswith("+00:00")


def test_unknown_id_is_404(client):
    assert client.get("/complaints/does-not-exist").status_code == 404


@pytest.mark.parametrize("field,value", [
    ("phone_number", "12345"),
    ("phone_number", "98765-43210"),
    ("email", "not-an-email"),
    ("name", "   "),
    ("complaint_details", ""),
])
def test_invalid_payloads_are_422(client, field, value):
    payload = {**VALID, field: value}
    assert client.post("/complaints", json=payload).status_code == 422


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["db"].endswith("test.db")
