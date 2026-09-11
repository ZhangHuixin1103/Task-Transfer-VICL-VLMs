from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from ..base import (
    ComparisonAdapter,
    InferenceResult,
    VICLSample,
    load_dataset_records,
    select_dataset_records,
    vicl_sample_from_records,
)
from ..metrics import StageTimer
from .common import import_from_root, resolve_model_reference, torch_dtype, working_directory


class LVMAdapter(ComparisonAdapter):
    """Official LVM visual-sentence inference with one same-task example."""

    name = "lvm"
    protocol = "[same-task demo input, demo output, query] -> query output"

    def __init__(
        self,
        repository: Path,
        dataset_json: Path,
        data_root: Path,
        checkpoint: str,
        vqvae_checkpoint: str = "Emma02/vqvae_ckpts",
        sample_index: int = 0,
        device: str = "cuda",
        dtype: str = "fp16",
        resolution: int = 256,
        seed: int = 42,
        demo_input: str | None = None,
        demo_output: str | None = None,
    ):
        if not device.startswith("cuda"):
            raise ValueError("The released LVM inference path requires CUDA")
        if torch_dtype(dtype) != torch.float16:
            raise ValueError("The released LVM GPU environment uses FP16 inference")
        if resolution != 256:
            raise ValueError("The released LVM VQ tokenizer uses 256x256 images")

        self.repository = repository.resolve()
        self.dataset_json = dataset_json.resolve()
        self.data_root = data_root.resolve()
        self.checkpoint = resolve_model_reference(checkpoint, self.repository)
        self.vqvae_checkpoint = resolve_model_reference(
            vqvae_checkpoint, self.repository
        )
        self.sample_index = sample_index
        self.device = device
        self.dtype = torch.float16
        self.resolution = resolution
        self.seed = seed
        self.context_frames = 16
        self.n_new_frames = 1
        self.n_candidates = 1
        self.temperature = 1.0
        self.top_p = 1.0
        self.demo_input = demo_input
        self.demo_output = demo_output
        self.records: list[dict[str, Any]] = []
        self._record_cache: dict[Path, list[dict[str, Any]]] = {}
        self.sample: VICLSample | None = None
        self.runtime = None
        self.read_image_to_tensor = None

    @property
    def conditions(self) -> Iterable[str]:
        return ("official",)

    def setup(self) -> None:
        initial_index = self.sample_index
        self.configure_samples(
            self.dataset_json,
            demo_input=self.demo_input,
            demo_output=self.demo_output,
        )
        self.select_sample(initial_index)

        evaluation_root = self.repository / "evaluation"
        inference = import_from_root("vqlm_demo.inference", evaluation_root)
        utilities = import_from_root("vqlm_demo.utils", evaluation_root)
        vqvae = import_from_root("vqlm_demo.vqvae_muse", evaluation_root)
        self.read_image_to_tensor = utilities.read_image_to_tensor

        # The released helper fixes the tokenizer location inside the source tree.
        # Redirect only that asset lookup so weights can remain in VICL_WEIGHTS;
        # model construction and generation still execute the official code.
        released_get_tokenizer = inference.get_tokenizer_muse

        def load_configured_tokenizer():
            return vqvae.VQGANModel.from_pretrained(self.vqvae_checkpoint)

        inference.get_tokenizer_muse = load_configured_tokenizer
        try:
            with working_directory(self.repository):
                self.runtime = inference.LocalInferenceModel(
                    checkpoint=self.checkpoint,
                    dtype="float16",
                    torch_device=self.device,
                    context_frames=self.context_frames,
                    use_lock=False,
                )
        finally:
            inference.get_tokenizer_muse = released_get_tokenizer

    def _context_image(self, relative_path: str) -> np.ndarray:
        if self.read_image_to_tensor is None:
            raise RuntimeError("LVM adapter is not set up")
        return self.read_image_to_tensor(str(self.data_root / relative_path))

    def run(self, condition: str) -> InferenceResult:
        if condition != "official" or self.sample is None or self.runtime is None:
            raise ValueError(condition)
        context_paths = (
            self.sample.task_a_input,
            self.sample.task_a_output,
            self.sample.task_b_input,
        )
        if len(set(context_paths)) != len(context_paths):
            raise ValueError(
                "LVM requires a demonstration pair and query with distinct image paths"
            )

        stages: Dict[str, Any] = {}
        with StageTimer(stages, "preprocess"):
            context = np.stack(
                [self._context_image(path) for path in context_paths], axis=0
            )
        with StageTimer(stages, "model_forward"):
            generated = self.runtime.generate_once(
                context,
                n_new_frames=self.n_new_frames,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        with StageTimer(stages, "postprocess"):
            image = np.asarray(generated[0], dtype=np.float32)
            pixels = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
            output = Image.fromarray(pixels)
        return InferenceResult(
            output=output,
            stage_seconds=stages,
            metadata={
                "output_size": list(output.size),
                "input_direction": self.protocol,
                "sample": self.sample.as_dict(),
                "n_candidates": self.n_candidates,
            },
        )

    def parameter_components(self, condition: str) -> Mapping[str, Any]:
        if condition != "official" or self.runtime is None:
            raise ValueError(condition)
        return {
            "lvm_autoregressive_model": self.runtime.model,
            "lvm_vqgan": self.runtime.tokenizer,
        }

    def configure_samples(
        self,
        dataset_json: Path,
        demo_input: str | None = None,
        demo_output: str | None = None,
        record_indices: Sequence[int] | None = None,
    ) -> None:
        self.dataset_json = dataset_json.resolve()
        self.demo_input = demo_input
        self.demo_output = demo_output
        source_records = self._record_cache.get(self.dataset_json)
        if source_records is None:
            source_records = load_dataset_records(self.dataset_json)
            self._record_cache[self.dataset_json] = source_records
        self.records = select_dataset_records(source_records, record_indices)
        self.select_sample(0)

    def sample_count(self) -> int:
        return len(self.records)

    def select_sample(self, sample_index: int) -> None:
        self.sample_index = sample_index
        self.sample = vicl_sample_from_records(
            self.records,
            sample_index,
            demo_input=self.demo_input,
            demo_output=self.demo_output,
            source=str(self.dataset_json),
        )

    def text_conditioning_metadata(self) -> Dict[str, Any]:
        return {
            "uses_text": False,
            "manifest_instruction_ignored": True,
            "reason": "The released LVM accepts only a visual sentence.",
        }

    def condition_metadata(self, condition: str) -> Dict[str, Any]:
        if condition != "official":
            raise ValueError(condition)
        return {
            "condition": condition,
            "checkpoint": self.checkpoint,
            "vqvae_checkpoint": self.vqvae_checkpoint,
            "device": self.device,
            "dtype": str(self.dtype),
            "native_resolution": [self.resolution, self.resolution],
            "context_frames": self.context_frames,
            "n_new_frames": self.n_new_frames,
            "n_candidates": self.n_candidates,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "sample": self.sample.as_dict() if self.sample else None,
            "protocol_note": (
                "The official LVM visual-sentence order, VQ tokenizer, FP16 model, "
                "256x256 resolution, temperature=1.0, and top_p=1.0 are preserved. "
                "One same-task demonstration is used to match the unified competitor "
                "input, and exactly one image is sampled per query. No task text or "
                "additional annotation is supplied."
            ),
        }

    def close(self) -> None:
        self.runtime = None
        self.read_image_to_tensor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
