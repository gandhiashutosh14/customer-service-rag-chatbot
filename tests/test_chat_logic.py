from chat_logic import (
    FIELDS,
    SlotFiller,
    format_complaint,
    format_ts,
    normalise_intent,
    parse_complaint_id,
    valid_email,
    valid_name,
    valid_phone,
)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------
def test_valid_name_accepts_letters_and_spaces_only():
    assert valid_name("Priya Sharma")
    assert not valid_name("A1")
    assert not valid_name("x")
    assert not valid_name("O'Brien")


def test_valid_name_rejects_text_that_looks_like_a_complaint():
    assert not valid_name("my order is late")


def test_valid_phone():
    assert valid_phone("9876543210")
    assert valid_phone("+919876543210")
    assert not valid_phone("12345")
    assert not valid_phone("98765-43210")


def test_valid_email():
    assert valid_email("user@example.com")
    assert not valid_email("user@example")
    assert not valid_email("not an email")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def test_parse_complaint_id_finds_uuid_in_sentence():
    text = "show details for complaint 3F2504E0-4F89-11D3-9A0C-0305E82C3301 please"
    assert parse_complaint_id(text) == "3f2504e0-4f89-11d3-9a0c-0305e82c3301"


def test_parse_complaint_id_returns_none_without_uuid():
    assert parse_complaint_id("show my complaint ABC123") is None


def test_normalise_intent_accepts_known_labels_and_falls_back():
    assert normalise_intent("file_complaint") == "file_complaint"
    assert normalise_intent("Intent: retrieve_complaint") == "general_query"  # label must be bare
    assert normalise_intent("retrieve_complaint\nextra text") == "retrieve_complaint"
    assert normalise_intent("something else") == "general_query"
    assert normalise_intent("") == "general_query"


def test_format_ts_pretty_prints_iso_and_passes_through_garbage():
    assert format_ts("2025-05-23T14:05:00") == "May 23, 2025 at 02:05 PM"
    assert format_ts("not a date") == "not a date"


def test_format_complaint_renders_every_field():
    text = format_complaint({
        "complaint_id": "abc", "name": "Priya", "phone_number": "9876543210",
        "email": "p@example.com", "complaint_details": "late delivery", "created_at": "2025-05-23T14:05:00",
    })
    for needle in ("abc", "Priya", "9876543210", "p@example.com", "late delivery", "May 23, 2025"):
        assert needle in text


# ---------------------------------------------------------------------------
# Slot filling state machine
# ---------------------------------------------------------------------------
def test_slot_filler_happy_path_collects_all_fields_in_order():
    f = SlotFiller(issue="a late delivery")
    assert "late delivery" in f.start()
    assert f.current_field == "name"

    ok, reply = f.submit("Priya Sharma")
    assert ok and "Priya Sharma" in reply and f.current_field == "phone_number"

    ok, reply = f.submit("9876543210")
    assert ok and "email" in reply.lower()

    ok, reply = f.submit("priya@example.com")
    assert ok and "late delivery" in reply

    ok, reply = f.submit("The parcel arrived four days late and the box was damaged.")
    assert ok and reply == ""
    assert f.complete
    assert list(f.data) == FIELDS
    assert f.data["email"] == "priya@example.com"


def test_slot_filler_rejects_invalid_input_and_stays_on_the_same_field():
    f = SlotFiller()
    f.start()
    ok, reply = f.submit("my order is late")
    assert not ok and "isn't valid" in reply
    assert f.current_field == "name"

    f.submit("Priya Sharma")
    ok, reply = f.submit("123")
    assert not ok and "Invalid phone number" in reply
    assert f.current_field == "phone_number"


def test_slot_filler_current_prompt_tracks_the_field_being_collected():
    f = SlotFiller(issue="a refund")
    assert f.current_prompt() == f.start()
    f.submit("Priya Sharma")
    assert f.current_prompt() == "Thank you, Priya Sharma. What is your phone number?"


def test_slot_filler_ignores_input_after_completion():
    f = SlotFiller()
    for value in ("Priya Sharma", "9876543210", "priya@example.com", "details"):
        f.submit(value)
    assert f.complete
    assert f.submit("anything") == (False, "")
