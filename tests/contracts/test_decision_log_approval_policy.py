"""Regression coverage for full-run approval policy decisions."""

from schemas.artifacts import validate_artifact


def test_decision_log_accepts_approval_policy_category():
    decision_log = {
        "version": "1.0",
        "project_id": "approved-full-run",
        "decisions": [
            {
                "decision_id": "d-approval-1",
                "stage": "proposal",
                "category": "approval_policy",
                "subject": "Human approval policy",
                "options_considered": [
                    {
                        "option_id": "full-run",
                        "label": "Full-run pre-authorization",
                        "score": 1.0,
                        "reason": "The user explicitly approved all later gates.",
                    }
                ],
                "selected": "full-run",
                "reason": "Record the explicit authorization required by the agent guide.",
                "user_visible": True,
                "user_approved": True,
                "confidence": 1.0,
            }
        ],
    }

    validate_artifact("decision_log", decision_log)
