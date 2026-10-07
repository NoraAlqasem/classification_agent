from __future__ import annotations

import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    from torchvision import models
    from tools import pretrained_weights
except ImportError as exc:
    IMPORT_ERROR = str(exc)
else:
    IMPORT_ERROR = ""


@unittest.skipIf(bool(IMPORT_ERROR), f"Pretrained weights dependencies unavailable: {IMPORT_ERROR}")
class PretrainedWeightsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="resnet-pretrained-catalog-")
        cls.root = Path(cls.temporary.name)
        cls.actual_checkpoint = cls.root / "actual_resnet18.pth"
        # One real backbone provides a normal serialized state dict; the
        # expanded scalar fixtures below avoid writing several large models.
        cls.actual_state = models.resnet18(weights=None).state_dict()
        torch.save(cls.actual_state, cls.actual_checkpoint)
        with torch.device("meta"):
            cls.templates = {
                arch: getattr(models, arch)(weights=None).state_dict()
                for arch in pretrained_weights.SUPPORTED_ARCHITECTURES
            }

    @classmethod
    def tearDownClass(cls):
        cls.actual_state = None
        cls.templates = None
        cls.temporary.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="resnet-pretrained-choices-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def compact_state(self, arch):
        zero_by_dtype = {}
        state = OrderedDict()
        for key, value in self.templates[arch].items():
            scalar = zero_by_dtype.setdefault(value.dtype, torch.zeros((), dtype=value.dtype))
            state[key] = scalar.expand(value.shape)
        return state

    def save(self, name, state):
        path = self.directory / name
        torch.save(state, path)
        return path

    def test_real_backbone_is_read_using_weights_only_and_meta(self):
        cuda_initialized = torch.cuda.is_initialized()
        with patch.object(pretrained_weights.torch, "load", wraps=torch.load) as load:
            candidates = pretrained_weights.discover_pretrained_weights(self.root)
        self.assertEqual(candidates, [{"arch": "resnet18", "weights": str(self.actual_checkpoint.resolve())}])
        load.assert_called_once_with(self.actual_checkpoint.resolve(), weights_only=True, map_location="meta")
        self.assertEqual(torch.cuda.is_initialized(), cuda_initialized)

    def test_all_supported_architectures_and_versioned_names_are_inferred_from_shapes(self):
        expected = []
        for arch in pretrained_weights.SUPPORTED_ARCHITECTURES:
            path = self.save(f"imagenet-{arch}-v2.pt", self.compact_state(arch))
            expected.append({"arch": arch, "weights": str(path.resolve())})
        self.assertEqual(pretrained_weights.discover_pretrained_weights(self.directory), expected)

    def test_misleading_filename_cannot_choose_wrong_architecture(self):
        path = self.save("resnet50.pth", self.compact_state("resnet18"))
        self.assertEqual(pretrained_weights.discover_pretrained_weights(self.directory),
                         [{"arch": "resnet18", "weights": str(path.resolve())}])

    def test_missing_extra_and_wrong_shape_tensors_are_rejected(self):
        state = self.compact_state("resnet18")
        missing = dict(state)
        missing.pop("fc.bias")
        extra = {**state, "unknown": torch.zeros(1)}
        wrong_shape = {**state, "fc.bias": torch.zeros(5)}
        for name, invalid in (("missing.pth", missing), ("extra.pth", extra), ("wrong.pth", wrong_shape)):
            self.save(name, invalid)
        with self.assertLogs(pretrained_weights.logger, level="WARNING") as warnings:
            self.assertEqual(pretrained_weights.discover_pretrained_weights(self.directory), [])
        self.assertEqual(len(warnings.records), 3)

    def test_legacy_batchnorm_buffers_are_accepted_only_when_trainer_can_supply_them(self):
        legacy = OrderedDict((key, value) for key, value in self.compact_state("resnet18").items()
                             if not key.endswith(".num_batches_tracked"))
        accepted = self.save("legacy.pth", legacy)
        modern_metadata = OrderedDict(legacy)
        modern_metadata._metadata = self.actual_state._metadata
        self.save("broken-modern.pth", modern_metadata)
        legacy_meta = OrderedDict((key, value.to("meta")) for key, value in legacy.items())
        modern_meta = OrderedDict(legacy_meta)
        modern_meta._metadata = modern_metadata._metadata
        with torch.device("meta"):
            self.assertFalse(models.resnet18(weights=None).load_state_dict(legacy_meta, strict=True).missing_keys)
            with self.assertRaises(RuntimeError):
                models.resnet18(weights=None).load_state_dict(modern_meta, strict=True)
        with self.assertLogs(pretrained_weights.logger, level="WARNING") as warnings:
            candidates = pretrained_weights.discover_pretrained_weights(self.directory)
        self.assertEqual(candidates, [{"arch": "resnet18", "weights": str(accepted.resolve())}])
        self.assertEqual(len(warnings.records), 1)

    def test_wrapped_prefixed_and_non_tensor_state_dicts_are_rejected(self):
        state = self.compact_state("resnet18")
        invalids = {
            "wrapper.pth": {"state_dict": state},
            "ddp.pth": {f"module.{key}": value for key, value in state.items()},
            "metadata.pth": {**state, "arch": "resnet18"},
            "empty.pth": {},
        }
        for name, invalid in invalids.items():
            self.save(name, invalid)
        with self.assertLogs(pretrained_weights.logger, level="WARNING"):
            self.assertEqual(pretrained_weights.discover_pretrained_weights(self.directory), [])

    def test_corrupt_or_unsupported_files_do_not_hide_valid_candidates(self):
        valid = self.save("valid.pth", self.compact_state("resnet18"))
        (self.directory / "corrupt.pt").write_text("not a checkpoint", encoding="utf-8")
        with torch.device("meta"):
            unsupported = models.resnet101(weights=None).state_dict()
        self.save("resnet101.pth", unsupported)
        with self.assertLogs(pretrained_weights.logger, level="WARNING") as warnings:
            candidates = pretrained_weights.discover_pretrained_weights(self.directory)
        self.assertEqual(candidates, [{"arch": "resnet18", "weights": str(valid.resolve())}])
        self.assertEqual(len(warnings.records), 2)

    def test_external_symlink_is_ignored_without_loading_the_target(self):
        escaped = self.directory / "outside.pth"
        escaped.symlink_to(self.actual_checkpoint)
        with patch.object(pretrained_weights.torch, "load") as load, \
             self.assertLogs(pretrained_weights.logger, level="WARNING"):
            self.assertEqual(pretrained_weights.discover_pretrained_weights(self.directory), [])
        load.assert_not_called()

    def test_internal_symlink_is_allowed_and_duplicate_targets_are_inspected_once(self):
        actual = self.save("original.pt", self.compact_state("resnet34"))
        (self.directory / "alias.pth").symlink_to(actual)
        with patch.object(pretrained_weights.torch, "load", wraps=torch.load) as load:
            candidates = pretrained_weights.discover_pretrained_weights(self.directory)
        self.assertEqual(candidates, [{"arch": "resnet34", "weights": str(actual.resolve())}])
        load.assert_called_once()

    def test_scan_is_limited_to_checkpoint_files_in_immediate_directory(self):
        self.save("ignore.bin", self.compact_state("resnet18"))
        nested = self.directory / "nested"
        nested.mkdir()
        torch.save(self.compact_state("resnet18"), nested / "resnet18.pth")
        (self.directory / "folder.pth").mkdir()
        with patch.object(pretrained_weights.torch, "load") as load:
            self.assertEqual(pretrained_weights.discover_pretrained_weights(self.directory), [])
        load.assert_not_called()

    def test_empty_missing_and_not_directory_locations_are_handled(self):
        self.assertEqual(pretrained_weights.discover_pretrained_weights(self.directory), [])
        self.assertEqual(pretrained_weights.discover_pretrained_weights(self.directory / "missing"), [])
        plain_file = self.directory / "file.txt"
        plain_file.write_text("test")
        with self.assertLogs(pretrained_weights.logger, level="WARNING"):
            self.assertEqual(pretrained_weights.discover_pretrained_weights(plain_file), [])

    def test_default_directory_can_be_used_without_an_override(self):
        path = self.save("resnet50.pth", self.compact_state("resnet50"))
        with patch.object(pretrained_weights, "PRETRAINED_WEIGHTS_DIR", self.directory):
            self.assertEqual(pretrained_weights.discover_pretrained_weights(),
                             [{"arch": "resnet50", "weights": str(path.resolve())}])

    def test_cached_templates_use_meta_tensors_without_model_allocation(self):
        pretrained_weights._backbone_shapes.cache_clear()
        constructors, observed_devices = {}, []
        for arch in pretrained_weights.SUPPORTED_ARCHITECTURES:
            original = getattr(models, arch)

            def construct(*args, factory=original, **kwargs):
                model = factory(*args, **kwargs)
                observed_devices.extend(value.device.type for value in model.state_dict().values())
                return model

            constructors[arch] = patch.object(pretrained_weights.models, arch, side_effect=construct)
        with constructors["resnet18"] as model18, constructors["resnet34"] as model34, \
             constructors["resnet50"] as model50:
            shapes = pretrained_weights._backbone_shapes()
            self.assertEqual(pretrained_weights._backbone_shapes(), shapes)
        for constructor in (model18, model34, model50):
            constructor.assert_called_once_with(weights=None)
        self.assertTrue(observed_devices)
        self.assertEqual(set(observed_devices), {"meta"})
        pretrained_weights._backbone_shapes.cache_clear()


if __name__ == "__main__":
    unittest.main()
