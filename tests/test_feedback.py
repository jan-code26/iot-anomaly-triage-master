"""
Unit tests for feedback schemas — no database required.

We set DATABASE_URL before any backend import so database.py doesn't raise
a RuntimeError at module load time.
"""
import os

os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost/test_db")

import pytest  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from uuid import UUID  # noqa: E402

from backend.schemas import FeedbackRequest  # noqa: E402


ALERT_ID = "00000000-0000-0000-0000-000000000001"


def test_feedback_request_valid_false_positive():
    req = FeedbackRequest(alert_event_id=UUID(ALERT_ID), label="FALSE_POSITIVE")
    assert req.label == "FALSE_POSITIVE"
    assert req.override is False


def test_feedback_request_valid_true_positive():
    req = FeedbackRequest(alert_event_id=UUID(ALERT_ID), label="TRUE_POSITIVE")
    assert req.label == "TRUE_POSITIVE"


def test_feedback_request_valid_uncertain():
    req = FeedbackRequest(alert_event_id=UUID(ALERT_ID), label="UNCERTAIN")
    assert req.label == "UNCERTAIN"


def test_feedback_request_invalid_label():
    with pytest.raises(ValidationError):
        FeedbackRequest(alert_event_id=ALERT_ID, label="WRONG_LABEL")


def test_feedback_request_override_default_false():
    req = FeedbackRequest(alert_event_id=UUID(ALERT_ID), label="TRUE_POSITIVE")
    assert req.override is False


def test_feedback_request_override_explicit_true():
    req = FeedbackRequest(
        alert_event_id=UUID(ALERT_ID),
        label="UNCERTAIN",
        override=True,
    )
    assert req.override is True
