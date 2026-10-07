from __future__ import annotations

import json
import unittest

from agent.llm_log_migration import (
    IMPORTED_CONSOLE_MARKER, convert_legacy_log, structured_content,
)


def request(request_id="request-one", *, stage="optimization", attempt=1, prompt=None):
    if prompt is None:
        prompt = '[system]\nTune ResNet.\nUse only measured results.\n\n[human]\n{"accuracy":0.6,"class":"تويوتا"}'
    return (
        f"\n{'=' * 80}\nRequest ID: {request_id}\nStarted (UTC): 2026-10-06T15:00:00+00:00\n"
        f"Stage: {stage}\nAttempt: {attempt}\nSession: /runs/session-one\nModel: fixture-model\n\n"
        f"Request:\n{prompt}\n"
    )


def completion(request_id="request-one", *, kind="response", content='{"updated_config":{"epochs":30}}',
               stage="optimization"):
    marker = "Response:\n" if kind == "response" else "Error: "
    return (
        f"\n--- LLM {kind} | Request ID: {request_id} | Stage: {stage} ---\n"
        f"Finished (UTC): 2026-10-06T15:00:01+00:00\n{marker}{content}\n"
    )


class StructuredContentTests(unittest.TestCase):
    def test_json_objects_arrays_and_fenced_multilingual_json_are_decoded(self):
        self.assertEqual(structured_content('{"class":"تويوتا","correct":3}'),
                         {"class": "تويوتا", "correct": 3})
        self.assertEqual(structured_content('[{"type":"text","text":"مرحبا"}]'),
                         [{"type": "text", "text": "مرحبا"}])
        self.assertEqual(structured_content(' \n```json\n{"rationale":"زيد عدد العصور"}\n```\n '),
                         {"rationale": "زيد عدد العصور"})

    def test_plain_malformed_and_nonfinite_json_keep_exact_original_text(self):
        for raw in (' Plain response.\n\n ', '{"epochs":', '{"score":NaN}',
                    '{"score":Infinity}', '[1,-Infinity]', '{"score":1e999}',
                    '```json\n{"score":NaN}\n```',
                    'prefix {"epochs":30}', '```json\n{"epochs":30}\n``` trailing',
                    '42', 'true', '"A plain JSON string"'):
            with self.subTest(raw=raw):
                self.assertEqual(structured_content(raw), raw)

    def test_non_string_structured_content_is_returned_unchanged(self):
        content = [{"type": "text", "text": "Learn make.\nتعلم النوع"}]
        self.assertIs(structured_content(content), content)
        dictionary = {"score": 0.6}
        self.assertIs(structured_content(dictionary), dictionary)
        self.assertIsNone(structured_content(None))


class LegacyLogMigrationTests(unittest.TestCase):
    def test_complete_request_preserves_metadata_context_and_response(self):
        raw = ' \n```json\n{"updated_config":{"epochs":30},"rationale":"تعلم"}\n```\n\n '
        records = convert_legacy_log(request() + completion(content=raw))
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(list(record)[:5], ["stage", "attempt", "status", "response", "error"])
        self.assertEqual(record["stage"], "optimization")
        self.assertEqual(record["attempt"], 1)
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["response"], {"updated_config": {"epochs": 30}, "rationale": "تعلم"})
        self.assertEqual(record["raw_response"], raw)
        self.assertIsNone(record["error"])
        self.assertEqual(record["request_id"], "request-one")
        self.assertEqual(record["started_at"], "2026-10-06T15:00:00+00:00")
        self.assertEqual(record["finished_at"], "2026-10-06T15:00:01+00:00")
        self.assertEqual(record["session"], "/runs/session-one")
        self.assertEqual(record["model"], "fixture-model")
        self.assertEqual(record["request"]["messages"], [
            {"role": "system", "content": "Tune ResNet.\nUse only measured results."},
            {"role": "user", "content": {"accuracy": 0.6, "class": "تويوتا"}},
        ])
        json.dumps(records, allow_nan=False, ensure_ascii=False)

    def test_interleaved_requests_pair_by_id_and_keep_start_order(self):
        first_raw = "First response.\n" + "=" * 80 + "\nA separator inside content.\n\n"
        raw = (request("first", stage="evaluation") + request("second", attempt=2)
               + completion("second", content='{"epochs":40}')
               + completion("first", stage="evaluation", content=first_raw))
        records = convert_legacy_log(raw)
        self.assertEqual([record["request_id"] for record in records], ["first", "second"])
        self.assertEqual(records[0]["raw_response"], first_raw)
        self.assertEqual(records[1]["response"], {"epochs": 40})
        self.assertTrue(all(record["status"] == "success" for record in records))

    def test_errors_and_interrupts_are_paired_without_losing_multiline_messages(self):
        for error_type, status in (("RuntimeError", "error"), ("KeyboardInterrupt", "interrupted"),
                                   ("SystemExit", "interrupted"), ("InterruptedError", "interrupted")):
            with self.subTest(error_type=error_type):
                message = "Provider stopped: 413\nFurther error details.\n"
                record = convert_legacy_log(request() + completion(
                    kind="error", content=f"{error_type}: {message}"))[0]
                self.assertEqual(record["status"], status)
                self.assertEqual(record["error"], {"type": error_type, "message": message})
                self.assertIsNone(record["response"])

    def test_pending_and_partially_written_requests_remain_present(self):
        complete = request("waiting")
        partial = f"\n{'=' * 80}\nRequest ID: unfinished\nStarted (UTC): 2026-10-06"
        records = convert_legacy_log(complete + partial)
        self.assertEqual([record["request_id"] for record in records], ["waiting", "unfinished"])
        self.assertEqual(records[0]["request"]["messages"][-1]["content"]["accuracy"], 0.6)
        self.assertEqual(records[1]["started_at"], "2026-10-06")
        self.assertTrue(all(record["status"] == "pending" for record in records))

    def test_orphan_responses_errors_and_duplicate_completions_are_retained(self):
        raw = (completion("orphan", content='[{"verdict":"good"}]')
               + completion("failure", kind="error", content="ValueError: Bad setting")
               + completion("orphan", content="A second completion remains available."))
        records = convert_legacy_log(raw)
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["response"], [{"verdict": "good"}])
        self.assertEqual(records[1]["error"], {"type": "ValueError", "message": "Bad setting"})
        self.assertEqual(records[2]["response"], "A second completion remains available.")
        self.assertTrue(all(record["request"]["messages"] == [] for record in records))

    def test_arbitrary_prefix_and_entire_unparsed_logs_keep_original_text(self):
        prefix = "Earlier notes:\nA response saved manually.\n"
        records = convert_legacy_log(prefix + request() + completion())
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["status"], "unparsed")
        self.assertEqual(records[0]["raw_response"], prefix)
        self.assertEqual(records[1]["status"], "success")
        self.assertEqual(convert_legacy_log(prefix)[0]["raw_response"], prefix)
        self.assertEqual(convert_legacy_log("\n \n"), [])
        self.assertEqual(convert_legacy_log(""), [])

    def test_imported_console_excerpt_keeps_original_content_and_stable_id(self):
        excerpt = ("Requesting LLM optimization (one request)...\n"
                   "The model is severely underfitting after only 2 epochs.\n"
                   "Increase the make head loss weight.\n")
        raw = (f"\n{'=' * 80}\n{IMPORTED_CONSOLE_MARKER}\n"
               "Saved (UTC): 2026-10-06T14:00:00+00:00\nStage: optimization\n\n" + excerpt + "\n")
        records = convert_legacy_log(raw + request() + completion())
        self.assertEqual(len(records), 2)
        imported = records[0]
        self.assertEqual(imported["status"], "imported")
        self.assertEqual(imported["stage"], "optimization")
        self.assertEqual(imported["raw_response"], excerpt)
        self.assertEqual(imported["finished_at"], "2026-10-06T14:00:00+00:00")
        self.assertEqual(imported["request_id"], convert_legacy_log(raw)[0]["request_id"])

    def test_malformed_json_and_message_role_aliases_do_not_lose_content(self):
        prompt = "[ai]\nPrevious explanation.\n\n\n[human]\n{\"metrics\":\n\n "
        response = '{"updated_config":{"epochs":40},\n"rationale":"Increase'
        record = convert_legacy_log(request(prompt=prompt) + completion(content=response))[0]
        self.assertEqual(record["request"]["messages"], [
            {"role": "assistant", "content": "Previous explanation.\n"},
            {"role": "user", "content": '{"metrics":\n\n '},
        ])
        self.assertEqual(record["raw_response"], response)
        self.assertEqual(record["response"], response)

    def test_duplicate_request_ids_preserve_both_starts_and_pair_pending_in_order(self):
        raw = (request("same", attempt=1) + request("same", attempt=2)
               + completion("same", content="First.") + completion("same", content="Second."))
        records = convert_legacy_log(raw)
        self.assertEqual([record["attempt"] for record in records], [1, 2])
        self.assertEqual([record["response"] for record in records], ["First.", "Second."])


if __name__ == "__main__":
    unittest.main()
