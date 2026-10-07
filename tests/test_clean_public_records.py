"""A readable experiment view cannot replace independently saved test proof."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from memory.experiment_format import (
    deserialize_experiment, serialize_experiment, serialize_internal_experiment,
)
from memory.experiment_store import ExperimentStore, experiment_score, test_evaluation_error
from test_experiment_format_regressions import saved_test_experiment


PUBLIC_SECTIONS = {
    "experiment", "objective", "config", "data", "training", "validation", "test", "evaluation", "artifacts",
}


class CleanPublicRecordTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="resnet-clean-public-record-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "runs" / "resnet"
        self.store = ExperimentStore(self.root)

    def record(self):
        record = saved_test_experiment()
        run = self.root / "session_4" / "exp_01"
        record.update(run_dir=str(run), best_weights=str(run / "weights" / "best_model.pth"),
                      last_weights=str(run / "weights" / "last_model.pth"))
        record["config"].update(out=str(run.parent), name=run.name)
        record["metrics"].update(checkpoint=record["best_weights"], output_dir=str(run / "test_evaluation"))
        record["test_result"].update(checkpoint=record["best_weights"], output_dir=str(run / "test_evaluation"))
        record["test_result"]["metrics"].update(checkpoint=record["best_weights"], output_dir=str(run / "test_evaluation"))
        return record

    def best(self):
        return self.store.get_best_experiment(target_metric="mean_acc", decision_split="test", test_csv="/dataset/test.csv")

    def test_clean_public_view_has_only_requested_sections_and_cannot_create_test_proof(self):
        original = self.record()
        untouched = deepcopy(original)
        document = serialize_experiment(original, session_id=4)
        self.assertEqual(set(document), PUBLIC_SECTIONS)

        def keys(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    yield key
                    yield from keys(child)
            elif isinstance(value, list):
                for child in value:
                    yield from keys(child)

        forbidden = {"schema_version", "analysis_fields", "analysis_overrides", "inference_available",
                     "inference_overrides", "training_overrides", "config_paths", "primary_accuracy",
                     "accuracy_alias", "reported_accuracy", "is_snapshot", "test_evaluated", "full_coverage",
                     "matched_images", "evaluated_images", "nonfinite_loss", "nonfinite_accuracy"}
        self.assertEqual(forbidden.intersection(keys(document)), set())
        self.assertIsNone(experiment_score(document))
        diagnostic = deserialize_experiment(document)
        self.assertFalse(diagnostic.get("test_result"))
        self.assertIsNotNone(test_evaluation_error(diagnostic.get("metrics", {}), diagnostic.get("test_result"),
                                                 diagnostic.get("test_csv")))
        self.assertEqual(original, untouched)

    def test_saved_public_file_and_bound_internal_record_preserve_verified_score_and_weights(self):
        original = self.record()
        path = self.store.save_experiment(original)
        self.assertEqual(path, Path(original["run_dir"]) / "exp_1.json")
        public = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(set(public), PUBLIC_SECTIONS)
        self.assertIsNone(experiment_score(public))
        knowledge = json.loads((self.root / "knowledge_base.json").read_text(encoding="utf-8"))
        entry = knowledge["experiments"][0]
        self.assertEqual(entry["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(entry["record"]["test_result"], original["test_result"])
        loaded = self.store.load_all_experiments()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(experiment_score(loaded[0]), 0.7)
        self.assertEqual(loaded[0]["test_result"], original["test_result"])
        self.assertEqual(loaded[0]["best_weights"], original["best_weights"])
        self.assertEqual(loaded[0]["last_weights"], original["last_weights"])
        self.assertEqual(self.best()["best_weights"], original["best_weights"])

    def test_modified_public_view_cannot_reuse_previously_bound_internal_proof(self):
        original = self.record()
        path = self.store.save_experiment(original)
        public = json.loads(path.read_text(encoding="utf-8"))
        public["test"]["mean_accuracy"] = 0.99
        path.write_text(json.dumps(public, indent=2), encoding="utf-8")
        self.assertIsNone(self.best())
        self.assertTrue(all(experiment_score(record) is None for record in self.store.load_all_experiments()))

    def test_private_rich_replacement_does_not_bypass_public_checksum_binding(self):
        original = self.record()
        path = self.store.save_experiment(original)
        rich = serialize_internal_experiment(original, session_id=4)
        self.assertEqual(experiment_score(rich), 0.7)
        path.write_text(json.dumps(rich, indent=2), encoding="utf-8")
        self.assertIsNone(self.best())
        self.assertTrue(all(experiment_score(record) is None for record in self.store.load_all_experiments()))

    def test_modified_internal_config_cannot_reuse_unchanged_public_view(self):
        original = self.record()
        path = self.store.save_experiment(original)
        knowledge_path = self.root / "knowledge_base.json"
        knowledge = json.loads(knowledge_path.read_text(encoding="utf-8"))
        entry = knowledge["experiments"][0]
        entry["record"]["config"]["epochs"] = 999
        self.assertEqual(experiment_score(entry["record"]), 0.7)
        self.assertEqual(entry["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        knowledge_path.write_text(json.dumps(knowledge, indent=2), encoding="utf-8")
        # Matching file bytes also require the stored record's public projection
        # to match, preventing a different configuration from reusing its score.
        self.assertIsNone(self.best())
        self.assertTrue(all(experiment_score(record) is None for record in self.store.load_all_experiments()))

    def test_partial_independent_test_proof_is_saved_without_creating_a_verified_best(self):
        original = self.record()
        raw = original["test_result"]["metrics"]
        raw["full_coverage"] = False
        raw["evaluated_images"] = 9
        raw["heads"]["make"].update(count=9, coverage=0.9)
        self.assertIsNone(experiment_score(original))
        path = self.store.save_experiment(original)
        self.assertTrue(path.is_file())
        loaded = self.store.load_all_experiments()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["test_result"], original["test_result"])
        self.assertIsNone(experiment_score(loaded[0]))
        self.assertIsNone(self.best())

    def test_missing_public_file_does_not_resurrect_stale_internal_record(self):
        path = self.store.save_experiment(self.record())
        path.unlink()
        self.assertEqual(self.store.load_all_experiments(), [])
        self.assertIsNone(self.best())

    def test_nonfinite_loss_stays_rejected_when_public_and_internal_json_are_cleaned(self):
        original = self.record()
        original["metrics"]["heads"]["make"]["loss"] = float("inf")
        self.assertIsNone(experiment_score(original))
        self.store.save_experiment(original)
        knowledge = json.loads((self.root / "knowledge_base.json").read_text(encoding="utf-8"))
        stored = knowledge["experiments"][0]["record"]
        self.assertTrue(stored["metrics"]["nonfinite_loss"])
        self.assertIsNone(stored["metrics"]["heads"]["make"]["loss"])
        self.assertIsNone(experiment_score(stored))
        self.assertIsNone(self.best())


if __name__ == "__main__":
    unittest.main()
