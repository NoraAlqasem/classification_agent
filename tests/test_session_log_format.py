"""Keep exactly the requested session, experiment, status and response fields."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from agent.session_log_format import sanitize_session_log, session_log_document, session_log_records


def request(request_id, attempt=1, stage="evaluation", model="fixture-model", started_at=None):
    return {
        "request_id": request_id, "attempt": attempt, "stage": stage,
        "status": "success", "response": {"reasoning": f"Result for {request_id}"},
        "request": {"messages": [{"role": "user", "content": {"accuracy": 0.7}}]},
        "error": None, "raw_response": '{"accuracy":0.7}', "model": model,
        "session": "/tmp/session_1", "started_at": started_at, "finished_at": None,
    }


class SessionLogFormatTests(unittest.TestCase):
    def test_projection_has_exact_requested_keys_and_keeps_native_response_values(self):
        records = [request("evaluate-1"), request("optimize-1", stage="optimization"),
                   request("evaluate-2", attempt=2), request("report-2", attempt=2, stage="report")]
        records[1]["response"] = {"updated_config": {"epochs": 40}, "rationale": "More training"}
        original = copy.deepcopy(records)
        document = session_log_document(records, Path("/tmp/session_1"))
        expected = {"session": {"session_id": "session_1", "model": "fixture-model", "experiments": {
            "exp_01": {"attempt": 1, "stages": {
                "evaluation": {"status": "success", "response": {"reasoning": "Result for evaluate-1"}},
                "optimization": {"status": "success", "response": records[1]["response"]},
            }},
            "exp_02": {"attempt": 2, "stages": {
                "evaluation": {"status": "success", "response": {"reasoning": "Result for evaluate-2"}},
            }},
        }}}
        self.assertEqual(document, expected)
        restored = session_log_records(json.loads(json.dumps(document)))
        self.assertEqual(len(restored), 3)
        self.assertTrue(all(set(record) == {"attempt", "stage", "model", "status", "response"} for record in restored))
        document["session"]["experiments"]["exp_01"]["stages"]["evaluation"]["response"]["reasoning"] = "changed"
        self.assertEqual(records, original)

    def test_repeated_stage_keeps_only_its_latest_status_and_response(self):
        document = session_log_document([request("first"), request("second"), request("third")], "/tmp/session_1")
        stage = document["session"]["experiments"]["exp_01"]["stages"]["evaluation"]
        self.assertEqual(stage, {"status": "success", "response": {"reasoning": "Result for third"}})
        self.assertEqual(len(session_log_records(document)), 1)
        self.assertNotIn("previous_requests", json.dumps(document))
        self.assertNotIn("request_id", json.dumps(document))

    def test_legacy_duplicate_ids_do_not_add_request_history_or_debug_metadata(self):
        first, second = request("same-id"), request("same-id")
        first.update(legacy_entry_id="same-id:1", source="legacy_log")
        second.update(legacy_entry_id="same-id:2", source="legacy_log", response="Second legacy result")
        document = session_log_document([first, second], "/tmp/session_1")
        self.assertEqual(session_log_records(document), [{"attempt": 1, "stage": "evaluation", "model": "fixture-model",
                                                        "status": "success", "response": "Second legacy result"}])
        self.assertNotIn("same-id", json.dumps(document))
        self.assertNotIn("legacy_entry_id", json.dumps(document))
        self.assertNotIn('"source"', json.dumps(document))

    def test_reports_unknown_stages_and_nonpositive_attempts_are_omitted(self):
        records = [request("invalid-data-report", attempt=0, stage="report"),
                   request("legacy-excerpt", attempt=None, stage="unknown"),
                   request("later-report", attempt=2, stage="report"), request("no-attempt", attempt=0)]
        document = session_log_document(records, "/tmp/session_1")
        self.assertEqual(document, {"session": {"session_id": "session_1", "model": "fixture-model", "experiments": {}}})
        self.assertEqual(session_log_records(document), [])
        self.assertNotIn("exp_00", json.dumps(document))
        self.assertNotIn("exp_02", json.dumps(document))

    def test_session_model_tracks_latest_request_after_projection_and_roundtrip(self):
        first = request("first-evaluation", model="old-model", started_at="2026-10-06T12:00:00+00:00")
        optimization = request("optimization", stage="optimization", model="old-model",
                               started_at="2026-10-06T12:00:01+00:00")
        latest = request("latest-evaluation", model="new-model", started_at="2026-10-06T12:00:02+00:00")
        document = session_log_document([first, optimization, latest], "/tmp/session_1")
        self.assertEqual(document["session"]["model"], "new-model")
        self.assertEqual(session_log_document(session_log_records(document), "/tmp/session_1"), document)

    def test_old_document_is_cleaned_including_extra_response_fields(self):
        evaluation = {"verdict": "underfitting", "target_reached": False, "issues": ["Few epochs"],
                      "recommendations": ["More epochs"], "reasoning": "Two epochs completed"}
        optimization = {"updated_config": {"arch": "resnet34", "weights": "/tmp/resnet34.pth", "epochs": 40,
                                            "head_loss_weights": {"make": 1.5}}, "rationale": "Compare backbone"}
        old = {"schema_version": 1, "session": {"session_id": "session_1", "model": "fixture-model",
            "path": "/tmp/session_1", "stages": {"report": {"status": "success", "response": "Do not keep report"}},
            "experiments": {
                "exp_01": {"attempt": 1, "timestamp": "old-time", "stages": {
                    "evaluation": {"status": "success", "response": {**evaluation, "usage": 20, "confidence": 0.8},
                                   "raw_response": "old raw", "request_id": "old-id", "request": {"messages": []}},
                    "optimization": {"status": "success", "response": {**optimization, "debug": "Remove"},
                                     "error": None, "previous_requests": [request("previous")]},
                    "report": {"status": "success", "response": "Narrative"},
                }},
                "exp_02": {"attempt": 2, "stages": {"report": {"status": "success", "response": "Only report"}}},
            }}}
        original = copy.deepcopy(old)
        clean = sanitize_session_log(old)
        self.assertEqual(clean, {"session": {"session_id": "session_1", "model": "fixture-model", "experiments": {
            "exp_01": {"attempt": 1, "stages": {
                "evaluation": {"status": "success", "response": evaluation},
                "optimization": {"status": "success", "response": optimization},
            }}}}})
        self.assertEqual(old, original)
        self.assertEqual(sanitize_session_log(clean), clean)

    def test_nonobject_actual_responses_remain_as_the_response_value(self):
        for response in (None, "{malformed JSON", ["native", "list"], 0):
            with self.subTest(response=response):
                record = request("result")
                record["response"] = response
                document = session_log_document([record], "/tmp/session_1")
                self.assertEqual(session_log_records(document)[0]["response"], response)

    def test_malformed_required_structure_is_rejected_without_inventing_values(self):
        valid = session_log_document([request("existing")], "/tmp/session_1")
        invalid = [None, [], {}, {"session": []}, {"session": {"session_id": "session_1", "experiments": []}}]
        wrong_attempt = copy.deepcopy(valid)
        wrong_attempt["session"]["experiments"]["exp_01"]["attempt"] = 2
        invalid.append(wrong_attempt)
        wrong_payload = copy.deepcopy(valid)
        wrong_payload["session"]["experiments"]["exp_01"]["stages"]["evaluation"] = []
        invalid.append(wrong_payload)
        missing_status = copy.deepcopy(valid)
        missing_status["session"]["experiments"]["exp_01"]["stages"]["evaluation"].pop("status")
        invalid.append(missing_status)
        for document in invalid:
            with self.subTest(document=document), self.assertRaises(ValueError):
                session_log_records(document)


if __name__ == "__main__":
    unittest.main()
