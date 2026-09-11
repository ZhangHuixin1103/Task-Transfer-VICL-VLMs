from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

import torch

from .adapters import T2TVICLAdapter, ToyAdapter
from .metrics import benchmark_callable, parameter_report, profile_flops
from .report import flatten_result, write_reports


T2T_NAMES = {
    "t2t-qwen": "qwen",
    "t2t-flux2": "flux2",
    "t2t-omnigen2": "omnigen2",
    "t2t-firered": "firered",
}
EXTERNAL_ROOT_ENV = "VICL_EXTERNAL_ROOT"


def _git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _repository_revision(path: Path) -> str | None:
    """Return a Git revision or the pinned commit for a vendored source tree."""
    resolved = path.resolve()
    for parent in (resolved, *resolved.parents):
        manifest_path = parent / "SOURCES.json"
        if not manifest_path.is_file():
            continue
        try:
            relative = resolved.relative_to(parent)
            source_name = relative.parts[0]
            document = json.loads(manifest_path.read_text(encoding="utf-8"))
            source = document["sources"][source_name]
            return str(source["commit"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
    return _git_revision(path)


def system_metadata(project_root: Path) -> Dict[str, Any]:
    cuda_devices = []
    cuda_device_properties = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(f"cuda:{index} {properties.name}")
            cuda_device_properties.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": [properties.major, properties.minor],
                    "multiprocessor_count": properties.multi_processor_count,
                }
            )
    package_versions = {}
    for package in (
        "accelerate",
        "bitsandbytes",
        "detectron2",
        "diffusers",
        "fairscale",
        "flash-attn",
        "k-diffusion",
        "numpy",
        "optimum",
        "pillow",
        "peft",
        "pytorch-lightning",
        "safetensors",
        "timm",
        "tokenizers",
        "transformers",
        "triton",
        "torchmetrics",
        "torchvision",
        "xformers",
    ):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    nvidia_smi = None
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,pstate,power.limit",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        nvidia_smi = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "cuda_devices": cuda_devices,
        "cuda_device_properties": cuda_device_properties,
        "nvidia_smi_index_name_uuid_driver_pstate_power_limit_w": nvidia_smi,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "allow_tf32_matmul": (
            torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else None
        ),
        "allow_tf32_cudnn": (
            torch.backends.cudnn.allow_tf32 if torch.cuda.is_available() else None
        ),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "package_versions": package_versions,
        "project_revision": _git_revision(project_root),
    }


def _resolve(path: Path, root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _external_repository(*parts: str) -> Path:
    configured = os.environ.get(EXTERNAL_ROOT_ENV)
    root = (
        Path(configured).expanduser().resolve()
        if configured
        else Path(__file__).resolve().parents[1] / "third_party"
    )
    if not root.is_dir():
        raise FileNotFoundError(
            f"External repository root does not name a directory: {root}. "
            f"Vendor sources under the project third_party directory or set "
            f"{EXTERNAL_ROOT_ENV}."
        )
    repository = root.joinpath(*parts)
    if not repository.is_dir():
        raise FileNotFoundError(
            f"External repository is missing below {EXTERNAL_ROOT_ENV}: {repository}"
        )
    return repository


def build_adapter(args, project_root: Path):
    dataset_json = _resolve(args.dataset_json, project_root)
    data_root = _resolve(args.data_root, project_root)

    if args.adapter in T2T_NAMES:
        conditions = args.conditions or ["fixed", "ours"]
        return T2TVICLAdapter(
            project_root=project_root,
            backend=T2T_NAMES[args.adapter],
            requested_conditions=conditions,
            dataset_json=dataset_json,
            data_root=data_root,
            sample_index=args.sample_index,
            model_id=args.model_id,
            prompt_checkpoint=args.prompt_checkpoint,
            prompt_base_model=args.prompt_base_model,
            device=args.device,
            dtype=args.dtype or "bf16",
            optimized=args.optimized,
            seed=args.seed if args.seed is not None else 42,
            steps=args.steps,
            resolution=args.resolution,
            demo_input=args.demo_input,
            demo_output=args.demo_output,
        )
    if args.adapter == "painter":
        from .adapters.external import PainterAdapter

        if not args.checkpoint:
            raise ValueError("--checkpoint is required for painter")
        return PainterAdapter(
            repository=_external_repository("Painter", "Painter"),
            dataset_json=dataset_json,
            data_root=data_root,
            sample_index=args.sample_index,
            checkpoint=args.checkpoint,
            device=args.device,
            dtype=args.dtype or "fp32",
            resolution=args.resolution or 448,
            task_protocol=args.painter_task,
            include_script_loss=args.painter_include_script_loss,
            demo_input=args.demo_input,
            demo_output=args.demo_output,
        )
    if args.adapter == "mae-vqgan":
        from .adapters.external import MAEVQGANAdapter

        if not args.checkpoint:
            raise ValueError("--checkpoint is required for mae-vqgan")
        return MAEVQGANAdapter(
            repository=_external_repository("MAE-VQGAN"),
            dataset_json=dataset_json,
            data_root=data_root,
            checkpoint=args.checkpoint,
            sample_index=args.sample_index,
            device=args.device,
            dtype=args.dtype or "fp32",
            architecture=args.mae_architecture,
            vqgan_config=args.vqgan_config,
            vqgan_checkpoint=args.vqgan_checkpoint,
            demo_input=args.demo_input,
            demo_output=args.demo_output,
        )
    if args.adapter == "prompt-gip":
        from .adapters.external import PromptGIPAdapter

        if not args.checkpoint:
            raise ValueError("--checkpoint is required for prompt-gip")
        return PromptGIPAdapter(
            repository=_external_repository("PromptGIP"),
            dataset_json=dataset_json,
            data_root=data_root,
            checkpoint=args.checkpoint,
            sample_index=args.sample_index,
            device=args.device,
            dtype=args.dtype or "fp32",
            architecture=args.promptgip_architecture,
            demo_input=args.demo_input,
            demo_output=args.demo_output,
        )
    if args.adapter == "visualcloze":
        from .adapters.external import VisualClozeAdapter

        if not args.checkpoint:
            raise ValueError("--checkpoint is required for visualcloze")
        return VisualClozeAdapter(
            repository=_external_repository("VisualCloze"),
            dataset_json=dataset_json,
            data_root=data_root,
            checkpoint=args.checkpoint,
            sample_index=args.sample_index,
            device=args.device,
            dtype=args.dtype or "bf16",
            resolution=args.resolution or 384,
            steps=30 if args.steps is None else args.steps,
            seed=args.seed if args.seed is not None else 0,
            text_prompt=args.text_prompt,
            demo_input=args.demo_input,
            demo_output=args.demo_output,
        )
    if args.adapter == "prompt-diffusion":
        from .adapters.external import PromptDiffusionAdapter

        if not args.checkpoint:
            raise ValueError("--checkpoint is required for prompt-diffusion")
        return PromptDiffusionAdapter(
            repository=_external_repository("Prompt-Diffusion"),
            dataset_json=dataset_json,
            data_root=data_root,
            checkpoint=args.checkpoint,
            sample_index=args.sample_index,
            device=args.device,
            dtype=args.dtype or "fp32",
            steps=100 if args.steps is None else args.steps,
            seed=args.seed if args.seed is not None else 1,
            content_prompt="",
            resolution=args.resolution or 512,
            config=args.prompt_diffusion_config,
            demo_input=args.demo_input,
            demo_output=args.demo_output,
        )
    if args.adapter == "instruct-diffusion":
        from .adapters.external import InstructDiffusionAdapter

        if not args.checkpoint:
            raise ValueError("--checkpoint is required for instruct-diffusion")
        return InstructDiffusionAdapter(
            repository=_external_repository("InstructDiffusion"),
            dataset_json=dataset_json,
            data_root=data_root,
            checkpoint=args.checkpoint,
            sample_index=args.sample_index,
            config=args.config,
            device=args.device,
            dtype=args.dtype or "fp16",
            resolution=args.resolution or 512,
            steps=100 if args.steps is None else args.steps,
            seed=args.seed if args.seed is not None else 42,
            text_prompt=args.text_prompt,
            cfg_text=args.cfg_text,
            cfg_image=args.cfg_image,
            fixed_square=args.shape_policy == "fixed-square",
            demo_input=args.demo_input,
            demo_output=args.demo_output,
        )
    if args.adapter == "lvm":
        from .adapters.lvm import LVMAdapter

        if not args.checkpoint:
            raise ValueError("--checkpoint is required for lvm")
        return LVMAdapter(
            repository=_external_repository("LVM"),
            dataset_json=dataset_json,
            data_root=data_root,
            checkpoint=args.checkpoint,
            vqvae_checkpoint=args.lvm_vqvae,
            sample_index=args.sample_index,
            device=args.device,
            dtype=args.dtype or "fp16",
            resolution=args.resolution or 256,
            seed=args.seed if args.seed is not None else 42,
            demo_input=args.demo_input,
            demo_output=args.demo_output,
        )
    if args.adapter == "toy":
        return ToyAdapter(device=args.device)
    raise ValueError(args.adapter)


def run_benchmark(args) -> Dict[str, Any]:
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("--warmup must be >= 0 and --repeats must be >= 1")
    if args.system_concurrency < 1:
        raise ValueError("--system-concurrency must be positive")
    if args.steps is not None and args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.resolution is not None and args.resolution < 1:
        raise ValueError("--resolution must be >= 1")
    project_root = Path(__file__).resolve().parents[1]
    adapter = build_adapter(args, project_root)
    print(f"Loading {adapter.name}...")
    adapter.setup()
    conditions = list(args.conditions or adapter.conditions)
    invalid = set(conditions) - set(adapter.conditions)
    if invalid:
        raise ValueError(
            f"Unsupported conditions for {adapter.name}: {sorted(invalid)}"
        )

    document: Dict[str, Any] = {
        "schema_version": 2,
        "adapter": adapter.name,
        "protocol": adapter.protocol,
        "system": system_metadata(project_root),
        "source_revision": _repository_revision(
            Path(getattr(adapter, "repository", project_root))
        ),
        "benchmark": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "batch_size": 1,
            "per_process_concurrency": 1,
            "declared_system_concurrency": args.system_concurrency,
            "concurrent_workloads_note": args.concurrent_workloads_note,
            "profile_flops": args.profile_flops,
            "parameters_only": args.parameters_only,
            "condition_resources_are_isolated": True,
            "condition_order": conditions,
            "flops_profiled_after_latency": True,
        },
        "conditions": [],
    }
    try:
        for condition in conditions:
            print(f"Benchmarking condition={condition}...")
            components = None
            adapter.prepare_condition(condition)
            try:
                components = adapter.parameter_components(condition)
                measurement_seed = getattr(adapter, "seed", None)
                if measurement_seed is None:
                    measurement_seed = args.seed if args.seed is not None else 42
                condition_result: Dict[str, Any] = {
                    "condition": condition,
                    "metadata": adapter.condition_metadata(condition),
                    "parameters": parameter_report(components),
                    "trained_parameters": adapter.trained_parameter_count(condition),
                    "latency": None,
                    "flops": None,
                }
                if not args.parameters_only:
                    condition_result["latency"] = benchmark_callable(
                        lambda condition=condition: adapter.run(condition),
                        warmup=args.warmup,
                        repeats=args.repeats,
                        seed=measurement_seed,
                    )
                    if args.profile_flops:
                        condition_result["flops"] = profile_flops(
                            lambda condition=condition: adapter.run(condition),
                            components=components,
                            top_k=args.flops_top_k,
                            seed=measurement_seed,
                        )
                document["conditions"].append(condition_result)
            finally:
                components = None
                adapter.release_condition(condition)
    finally:
        adapter.close()
    return document


def write_result(document: Dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_path = output_dir / f"{document['adapter']}_{stamp}.json"
    result_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    (output_dir / f"{document['adapter']}_latest.json").write_text(
        json.dumps(document, indent=2), encoding="utf-8"
    )
    write_reports(flatten_result(document), output_dir)
    return result_path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Measure logical parameters, runtime formula FLOPs, and inference latency"
    )
    result.add_argument(
        "--adapter",
        required=True,
        choices=[
            *T2T_NAMES,
            "painter",
            "mae-vqgan",
            "prompt-gip",
            "visualcloze",
            "prompt-diffusion",
            "instruct-diffusion",
            "lvm",
            "toy",
        ],
    )
    result.add_argument("--conditions", nargs="+")
    result.add_argument(
        "--dataset-json", type=Path, default=Path("data/dataset/eval_dataset.json")
    )
    result.add_argument("--data-root", type=Path, default=Path("data/tasks"))
    result.add_argument("--sample-index", type=int, default=0)
    result.add_argument("--demo-input")
    result.add_argument("--demo-output")
    result.add_argument("--model-id")
    result.add_argument("--checkpoint")
    result.add_argument("--vqgan-config")
    result.add_argument("--vqgan-checkpoint")
    result.add_argument(
        "--lvm-vqvae",
        default="Emma02/vqvae_ckpts",
        help="LVM VQGAN directory or Hugging Face model ID",
    )
    result.add_argument("--config", default="configs/instruct_diffusion.yaml")
    result.add_argument(
        "--prompt-diffusion-config", default="models/cldm_v15.yaml"
    )
    result.add_argument(
        "--prompt-checkpoint",
        default="Qwen3-VL/qwen-vl-finetune/output/checkpoint-4875",
    )
    result.add_argument("--prompt-base-model", default="Qwen/Qwen3-VL-4B-Instruct")
    result.add_argument("--device", default="cuda")
    result.add_argument("--dtype", choices=["fp32", "fp16", "bf16"])
    result.add_argument("--steps", type=int)
    result.add_argument("--resolution", type=int)
    result.add_argument(
        "--shape-policy",
        choices=["native-model", "fixed-square"],
        default="native-model",
    )
    result.add_argument(
        "--painter-task",
        choices=["restoration", "depth", "semantic", "discrete", "generic"],
        default="restoration",
    )
    result.add_argument("--painter-include-script-loss", action="store_true")
    result.add_argument(
        "--mae-architecture", default="mae_vit_large_patch16"
    )
    result.add_argument(
        "--promptgip-architecture",
        default="mae_vit_large_patch16_input256",
    )
    result.add_argument("--seed", type=int)
    result.add_argument(
        "--text-prompt",
        default="perform the demonstrated visual task",
        help=(
            "Task instruction for text-instruction adapters; Prompt-Diffusion "
            "intentionally uses an empty content prompt"
        ),
    )
    result.add_argument("--cfg-text", type=float, default=5.0)
    result.add_argument("--cfg-image", type=float, default=1.25)
    result.add_argument("--optimized", action="store_true")
    result.add_argument("--warmup", type=int, default=1)
    result.add_argument("--repeats", type=int, default=5)
    result.add_argument("--profile-flops", action="store_true")
    result.add_argument("--flops-top-k", type=int, default=20)
    result.add_argument("--parameters-only", action="store_true")
    result.add_argument(
        "--system-concurrency",
        type=int,
        default=1,
        help="Declared maximum number of model processes sharing the server during timing",
    )
    result.add_argument(
        "--concurrent-workloads-note",
        default="none",
        help="Free-text record of other jobs sharing GPUs/host resources",
    )
    result.add_argument(
        "--output-dir", type=Path, default=Path("comparison/outputs/resources")
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("--warmup must be >= 0 and --repeats must be >= 1")
    document = run_benchmark(args)
    project_root = Path(__file__).resolve().parents[1]
    output_dir = _resolve(args.output_dir, project_root)
    result_path = write_result(document, output_dir)
    print(f"Wrote {result_path}")
    print(f"Wrote {output_dir / 'comparison.csv'}")
    print(f"Wrote {output_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
