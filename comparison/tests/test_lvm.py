from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from comparison.benchmark import _repository_revision, parser as benchmark_parser
from comparison.quality import parser as quality_parser


TASK_TRANSFER = Path(__file__).resolve().parents[2]
LVM_ROOT = TASK_TRANSFER / "third_party/LVM"
LVM_COMMIT = "b6de939ef0eb1ee6593445a7f5268145f338749b"


class LVMInterfaceTest(unittest.TestCase):
    def test_official_source_and_cli_registration(self):
        self.assertEqual(_repository_revision(LVM_ROOT), LVM_COMMIT)
        tree = ast.parse(
            (LVM_ROOT / "evaluation/vqlm_demo/inference.py").read_text(
                encoding="utf-8"
            )
        )
        definitions = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        }
        self.assertIn("LocalInferenceModel", definitions)

        for parser in (benchmark_parser(), quality_parser()):
            choices = next(
                action.choices
                for action in parser._actions
                if action.dest == "adapter"
            )
            self.assertIn("lvm", choices)

    def test_official_defaults_and_one_shot_visual_sentence(self):
        from comparison.adapters.lvm import LVMAdapter

        class FakeRuntime:
            call = None

            def generate_once(self, context, **kwargs):
                self.call = (context.copy(), kwargs)
                return np.full((1, 256, 256, 3), 0.5, dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            colors = {
                "demo_input.png": (255, 0, 0),
                "demo_output.png": (0, 255, 0),
                "query.png": (0, 0, 255),
                "target.png": (255, 255, 255),
            }
            for filename, color in colors.items():
                Image.new("RGB", (16, 12), color).save(root / filename)
            dataset = root / "eval.json"
            dataset.write_text(
                json.dumps(
                    [{"image_path": "query.png", "target_path": "target.png"}]
                ),
                encoding="utf-8",
            )
            adapter = LVMAdapter(
                repository=LVM_ROOT,
                dataset_json=dataset,
                data_root=root,
                checkpoint="Emma02/LVM_ckpts",
                demo_input="demo_input.png",
                demo_output="demo_output.png",
            )
            adapter.configure_samples(
                dataset,
                demo_input="demo_input.png",
                demo_output="demo_output.png",
            )
            adapter.read_image_to_tensor = lambda path: np.asarray(
                Image.open(path).convert("RGB").resize((256, 256)),
                dtype=np.float32,
            ) / 255.0
            adapter.runtime = FakeRuntime()
            result = adapter.run("official")
            context, kwargs = adapter.runtime.call

        self.assertEqual(context.shape, (3, 256, 256, 3))
        self.assertTrue(np.allclose(context[:, 0, 0], np.eye(3)))
        self.assertEqual(kwargs["n_new_frames"], 1)
        self.assertEqual(kwargs["temperature"], 1.0)
        self.assertEqual(kwargs["top_p"], 1.0)
        self.assertEqual(result.output.size, (256, 256))
        self.assertEqual(adapter.n_candidates, 1)
        self.assertEqual(adapter.context_frames, 16)
        self.assertFalse(adapter.text_conditioning_metadata()["uses_text"])


if __name__ == "__main__":
    unittest.main()
