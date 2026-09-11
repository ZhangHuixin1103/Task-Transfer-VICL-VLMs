from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .benchmark import build_adapter, parser as benchmark_parser, system_metadata
from .base import load_dataset_records
from .datasets import (
    TaskSpec,
    load_task_manifest,
    matching_record_indices,
    select_indices,
)
from .metrics import (
    benchmark_dataset_callable,
    parameter_report,
    profile_flops,
    summarize_seconds,
)


def _select_tasks(tasks: list[TaskSpec], requested: list[str] | None) -> list[TaskSpec]:
    if not requested:
        return tasks
    names = set(requested)
    selected = [task for task in tasks if task.name in names]
    missing = names - {task.name for task in selected}
    if missing:
        raise ValueError(f"Unknown tasks: {sorted(missing)}")
    return selected


def _sampling_seed(base: int, task_name: str) -> int:
    return base + sum((index + 1) * ord(char) for index, char in enumerate(task_name))


def _select_warmup_indices(
    count: int,
    measured_indices: list[int],
    warmup: int,
    seed: int,
) -> tuple[list[int], str]:
    if warmup < 1:
        return [], "disabled"
    measured = set(measured_indices)
    candidates = [index for index in range(count) if index not in measured]
    if candidates:
        selected = select_indices(
            len(candidates), min(warmup, len(candidates)), seed + 1
        )
        return [candidates[index] for index in selected], "disjoint_from_measured"
    return measured_indices[: min(warmup, len(measured_indices))], (
        "reuses_measured_indices_because_the_full_split_is_measured"
    )


def _set_task_prompt(adapter, prompt: str, task_name: str) -> None:
    setter = getattr(adapter, "set_task_prompt", None)
    if callable(setter):
        setter(task_name, prompt)
    elif hasattr(adapter, "text_prompt"):
        adapter.text_prompt = prompt


def _select_and_prime_inputs(adapter, sample_index: int) -> None:
    """Select one query and remove cold filesystem reads from timed latency."""
    adapter.select_sample(sample_index)
    sample = getattr(adapter, "sample", None)
    data_root = getattr(adapter, "data_root", None)
    if sample is None or data_root is None:
        return
    for relative_path in dict.fromkeys(
        (sample.task_a_input, sample.task_a_output, sample.task_b_input)
    ):
        path = Path(data_root) / relative_path
        with path.open("rb") as handle:
            while handle.read(1024 * 1024):
                pass


def _suite_signature(adapter, args, manifest_path: Path, conditions: list[str]) -> dict[str, Any]:
    signature = {
        "adapter": adapter.name,
        "conditions": conditions,
        "task_manifest": str(manifest_path),
        "data_root": str(args.data_root),
        "sampling_seed": args.sampling_seed,
        "resolution": getattr(adapter, "resolution", args.resolution),
        "model_id": getattr(adapter, "model_id", None),
        "checkpoint": getattr(adapter, "checkpoint", None),
        "vqvae_checkpoint": getattr(adapter, "vqvae_checkpoint", None),
        "prompt_checkpoint": getattr(adapter, "prompt_checkpoint", None),
        "prompt_base_model": getattr(adapter, "prompt_base_model", None),
        "steps": getattr(adapter, "steps", None),
        "seed": getattr(adapter, "seed", None),
        "dtype": str(getattr(adapter, "dtype", None)),
        "optimized": getattr(adapter, "optimized", None),
        "config": str(getattr(adapter, "config_path", None)),
        "cfg_text": getattr(adapter, "cfg_text", None),
        "cfg_image": getattr(adapter, "cfg_image", None),
        "painter_task": getattr(adapter, "task_protocol", None),
    }
    text_conditioning = getattr(adapter, "text_conditioning_metadata", None)
    if callable(text_conditioning):
        signature["model_text_conditioning"] = text_conditioning()
    return signature


def _load_resume_document(path: Path | None, signature: dict[str, Any]) -> tuple[Path | None, dict[str, Any] | None]:
    if path is None:
        return None, None
    resolved = path.expanduser().resolve()
    document = json.loads(resolved.read_text(encoding="utf-8"))
    if document.get("kind") != "multi_task_comparison_suite":
        raise ValueError(f"--resume-from is not a multi-task suite JSON: {resolved}")
    if document.get("resume_signature") != signature:
        raise ValueError(
            "--resume-from model/task/seed/resolution configuration does not match "
            f"the current run: {resolved}"
        )
    return resolved, document


def _resume_task_latency(
    document: dict[str, Any] | None, condition: str, task: str
) -> dict[str, Any] | None:
    if document is None:
        return None
    condition_result = next(
        (item for item in document.get("conditions", []) if item.get("condition") == condition),
        None,
    )
    if condition_result is None:
        return None
    task_result = next(
        (item for item in condition_result.get("tasks", []) if item.get("task") == task),
        None,
    )
    return task_result.get("latency") if task_result is not None else None


def _maximum_memory_reports(*reports: dict[str, Any] | None, key: str) -> dict[str, int]:
    merged: dict[str, int] = {}
    for report in reports:
        for device, value in (report or {}).get(key, {}).items():
            merged[device] = max(merged.get(device, 0), int(value))
    return merged


def _merge_latency_reports(
    previous: dict[str, Any] | None,
    current: dict[str, Any] | None,
    target_indices: list[int],
    resume_source: Path | None,
    processing_order: str,
) -> dict[str, Any]:
    rows_by_index: dict[int, dict[str, Any]] = {}
    previous_indices = list((previous or {}).get("measured_indices", []))
    if not set(previous_indices).issubset(target_indices):
        raise ValueError(
            "--resume-from contains measured indices outside the current target set"
        )
    for report in (previous, current):
        for row in (report or {}).get("samples", []):
            index = int(row["sample_index"])
            if index in rows_by_index:
                raise ValueError(f"Duplicate resumed latency sample index: {index}")
            rows_by_index[index] = row
    missing = [index for index in target_indices if index not in rows_by_index]
    if missing:
        raise ValueError(f"Missing merged latency indices: {missing[:20]}")

    rows = [rows_by_index[index] for index in target_indices]
    stage_names = sorted(
        {name for row in rows for name in row.get("stages_s", {})}
    )
    template = current or previous or {}
    merged = dict(template)
    merged.update(
        {
            "end_to_end": summarize_seconds(
                [float(row["end_to_end_s"]) for row in rows]
            ),
            "stages": {
                name: summarize_seconds(
                    [
                        float(row["stages_s"][name])
                        for row in rows
                        if name in row.get("stages_s", {})
                    ]
                )
                for name in stage_names
            },
            "samples": rows,
            "measured_indices": target_indices,
            "resident_cuda_memory_bytes": _maximum_memory_reports(
                previous, current, key="resident_cuda_memory_bytes"
            ),
            "peak_cuda_memory_allocated_bytes": _maximum_memory_reports(
                previous, current, key="peak_cuda_memory_allocated_bytes"
            ),
            "peak_cuda_memory_reserved_bytes": _maximum_memory_reports(
                previous, current, key="peak_cuda_memory_reserved_bytes"
            ),
            "peak_cuda_memory_bytes": _maximum_memory_reports(
                previous, current, key="peak_cuda_memory_bytes"
            ),
            "resume": {
                "source": str(resume_source) if resume_source else None,
                "cross_process_reuse": previous is not None,
                "reused_indices": previous_indices,
                "newly_measured_indices": list(
                    (current or {}).get("measured_indices", [])
                ),
                "processing_order": processing_order,
                "warning": (
                    "Raw latency samples were combined across two model processes."
                    if previous is not None
                    else None
                ),
            },
        }
    )
    return merged


def _aggregate_tasks(tasks: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not tasks:
        return None
    task_means = [task["latency"]["end_to_end"]["mean_s"] for task in tasks]
    task_medians = [task["latency"]["end_to_end"]["median_s"] for task in tasks]
    pooled = [
        value
        for task in tasks
        for value in task["latency"]["end_to_end"]["samples_s"]
    ]
    flop_values = [
        task["flops"].get("estimated_total_flops")
        for task in tasks
        if task.get("flops") is not None and task["flops"].get("status") == "ok"
    ]
    complete_flop_values = [value for value in flop_values if value is not None]
    return {
        "definition": (
            "Macro values give every task equal weight. Pooled latency gives every "
            "measured query equal weight and is secondary when task sample counts differ."
        ),
        "tasks": len(tasks),
        "queries": len(pooled),
        "macro_task_mean_latency_s": statistics.fmean(task_means),
        "macro_task_median_latency_s": statistics.fmean(task_medians),
        "pooled_query_mean_latency_s": statistics.fmean(pooled),
        "pooled_query_median_latency_s": statistics.median(pooled),
        "macro_task_flops": (
            statistics.fmean(complete_flop_values)
            if len(complete_flop_values) == len(tasks)
            else None
        ),
        "flops_complete_for_all_tasks": len(complete_flop_values) == len(tasks),
    }


def _select_flop_indices(indices: list[int], limit: int) -> list[int]:
    if limit < 1:
        raise ValueError("--flops-samples-per-task must be positive")
    if limit >= len(indices):
        return list(indices)
    if limit == 1:
        return [indices[len(indices) // 2]]
    positions = [
        round(rank * (len(indices) - 1) / (limit - 1)) for rank in range(limit)
    ]
    return [indices[position] for position in positions]


def _aggregate_flop_profiles(
    sample_indices: list[int], profiles: list[dict[str, Any]]
) -> dict[str, Any]:
    if not profiles or len(sample_indices) != len(profiles):
        raise ValueError("FLOPs profiles and sample indices must be non-empty and aligned")
    values = [
        profile.get("estimated_total_flops")
        for profile in profiles
        if profile.get("estimated_total_flops") is not None
    ]
    all_complete = len(values) == len(profiles) and all(
        profile.get("status") == "ok" for profile in profiles
    )
    confidences = {profile.get("confidence") for profile in profiles}
    unsupported = [
        {"sample_index": index, **row}
        for index, profile in zip(sample_indices, profiles)
        for row in profile.get("unsupported_nontrivial_operators", [])
    ]
    formula_values = [
        profile.get("formula_flops")
        for profile in profiles
        if profile.get("formula_flops") is not None
    ]
    opaque_values = [
        profile.get("opaque_module_flops")
        for profile in profiles
        if profile.get("opaque_module_flops") is not None
    ]
    result = dict(profiles[0]) if len(profiles) == 1 else {}
    result.update(
        {
            "status": "ok" if all_complete else "partial",
            "confidence": (
                next(iter(confidences)) if len(confidences) == 1 else "mixed"
            ),
            "estimated_total_flops": statistics.fmean(values) if values else None,
            "estimated_total_flops_min": min(values) if values else None,
            "estimated_total_flops_max": max(values) if values else None,
            "profile_sample_count": len(profiles),
            "profile_sample_indices": sample_indices,
            "formula_flops": (
                statistics.fmean(formula_values) if formula_values else None
            ),
            "opaque_module_flops": (
                statistics.fmean(opaque_values) if opaque_values else None
            ),
            "unsupported_nontrivial_operators": unsupported,
            "profiles": [
                {"sample_index": index, **profile}
                for index, profile in zip(sample_indices, profiles)
            ],
            "definition": profiles[0].get("definition"),
            "scope": profiles[0].get("scope"),
            "convention": profiles[0].get("convention"),
            "flop_convention": profiles[0].get("flop_convention"),
            "aggregation_definition": (
                "Arithmetic mean over deterministic, evenly spaced measured-query indices. "
                "The aggregate is exact only when every per-sample profile has status=ok."
            ),
        }
    )
    return result


def run_suite(args) -> dict[str, Any]:
    if args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.steps is not None and args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.resolution is not None and args.resolution < 1:
        raise ValueError("--resolution must be positive")
    if args.flops_samples_per_task < 1:
        raise ValueError("--flops-samples-per-task must be positive")
    if args.system_concurrency < 1:
        raise ValueError("--system-concurrency must be positive")
    if args.adapter == "toy":
        raise ValueError(
            "The toy adapter is only for comparison.benchmark smoke tests, not the "
            "multi-task suite"
        )
    if (
        args.adapter
        in {"painter", "mae-vqgan", "prompt-gip", "visualcloze"}
        and not args.checkpoint
        and not args.parameters_only
    ):
        raise ValueError(
            f"{args.adapter} latency/FLOPs require the official checkpoint; use "
            "--parameters-only to inspect architecture parameters without weights"
        )

    project_root = Path(__file__).resolve().parents[1]
    default_manifest_name = (
        "t2t_tasks.json"
        if args.adapter.startswith("t2t-")
        else "competitor_tasks.json"
    )
    manifest_path = (
        args.task_manifest or Path(__file__).with_name(default_manifest_name)
    ).resolve()
    manifest, all_tasks = load_task_manifest(manifest_path)
    tasks = _select_tasks(all_tasks, args.tasks)
    if not tasks:
        raise ValueError("No tasks selected")
    cross_task_flags = {task.is_cross_task for task in tasks}
    if len(cross_task_flags) != 1:
        raise ValueError("A task manifest cannot mix cross-task and same-task entries")
    if True in cross_task_flags and not args.adapter.startswith("t2t-"):
        raise ValueError(
            "Cross-task records are reserved for T2T-VICL adapters; official "
            "baselines use the standardized same-task competitor manifest"
        )
    if False in cross_task_flags and args.adapter.startswith("t2t-"):
        raise ValueError(
            "T2T-VICL resource suites require the original cross-task manifest "
            "comparison/t2t_tasks.json"
        )

    default_data_root = project_root / manifest["data_root"]
    args.data_root = (args.data_root or default_data_root).resolve()
    first = tasks[0]
    args.dataset_json = args.data_root / first.eval_json
    args.demo_input = first.demo_input
    args.demo_output = first.demo_output
    args.text_prompt = first.text_prompt

    source_record_cache: dict[Path, list[dict[str, Any]]] = {}

    def task_records(task: TaskSpec) -> tuple[Path, list[int] | None, int]:
        json_path = (args.data_root / task.eval_json).resolve()
        records = source_record_cache.get(json_path)
        if records is None:
            records = load_dataset_records(json_path)
            source_record_cache[json_path] = records
        return json_path, matching_record_indices(records, task), len(records)

    adapter = build_adapter(args, project_root)
    setup_started = time.perf_counter()
    print(f"Loading {adapter.name} once for {len(tasks)} tasks...")
    adapter.setup()
    setup_seconds = time.perf_counter() - setup_started
    conditions = list(args.conditions or adapter.conditions)
    invalid = set(conditions) - set(adapter.conditions)
    if invalid:
        raise ValueError(f"Unsupported conditions for {adapter.name}: {sorted(invalid)}")
    resume_signature = _suite_signature(adapter, args, manifest_path, conditions)
    resume_path, resume_document = _load_resume_document(
        args.resume_from, resume_signature
    )

    document: dict[str, Any] = {
        "schema_version": 1,
        "kind": "multi_task_comparison_suite",
        "adapter": adapter.name,
        "protocol": adapter.protocol,
        "benchmark_family": manifest.get("benchmark_family"),
        "task_manifest": str(manifest_path),
        "controlled_protocol": {
            **manifest["controlled_protocol"],
            "resolution": getattr(adapter, "resolution", args.resolution),
            "resolution_policy": (
                "explicit --resolution override"
                if args.resolution is not None
                else "official/original adapter default"
            ),
            "warmup_queries": args.warmup,
            "measured_queries_per_task": args.max_samples,
            "sampling_seed": args.sampling_seed,
            "shape_policy": args.shape_policy,
        },
        "system": system_metadata(project_root),
        "model_setup_seconds": setup_seconds,
        "resume_signature": resume_signature,
        "benchmark": {
            "batch_size": 1,
            "per_process_concurrency": 1,
            "declared_system_concurrency": args.system_concurrency,
            "concurrent_workloads_note": args.concurrent_workloads_note,
            "profile_flops": args.profile_flops,
            "parameters_only": args.parameters_only,
            "condition_order": conditions,
            "model_loaded_once_across_tasks": True,
            "flops_profiled_after_latency": True,
            "input_file_cache_policy": (
                "Read the three model-input files immediately before timing so physical "
                "disk cold-start variance is excluded; image open/decode/resize remains timed."
            ),
            "flops_samples_per_task_for_variable_ours": args.flops_samples_per_task,
            "flops_samples_per_task_for_fixed_or_official": 1,
            "resume_from": str(resume_path) if resume_path else None,
            "reverse_order": args.reverse_order,
        },
        "conditions": [],
    }

    try:
        for condition in conditions:
            print(f"Preparing condition={condition}...")
            prepare_started = time.perf_counter()
            adapter.prepare_condition(condition)
            condition_prepare_seconds = time.perf_counter() - prepare_started
            components = None
            try:
                components = adapter.parameter_components(condition)
                condition_result: dict[str, Any] = {
                    "condition": condition,
                    "condition_prepare_seconds": condition_prepare_seconds,
                    "parameters": parameter_report(components),
                    "trained_parameters": adapter.trained_parameter_count(condition),
                    "tasks": [],
                }
                if args.parameters_only:
                    document["conditions"].append(condition_result)
                    continue

                for task in tasks:
                    json_path, record_indices, source_record_count = task_records(task)
                    print(f"Latency: condition={condition}, task={task.name}...")
                    adapter.configure_samples(
                        json_path,
                        demo_input=task.demo_input,
                        demo_output=task.demo_output,
                        record_indices=record_indices,
                    )
                    _set_task_prompt(adapter, task.text_prompt, task.name)
                    count = adapter.sample_count()
                    target_indices = select_indices(
                        count,
                        args.max_samples,
                        _sampling_seed(args.sampling_seed, task.name),
                    )
                    previous_latency = _resume_task_latency(
                        resume_document, condition, task.name
                    )
                    if resume_document is not None and previous_latency is None:
                        raise ValueError(
                            f"--resume-from has no latency rows for {condition}/{task.name}"
                        )
                    reused_indices = set(
                        (previous_latency or {}).get("measured_indices", [])
                    )
                    if not reused_indices.issubset(target_indices):
                        raise ValueError(
                            f"Resume indices for {condition}/{task.name} are not a "
                            "subset of the requested target indices"
                        )
                    remaining_indices = [
                        index for index in target_indices if index not in reused_indices
                    ]
                    processing_indices = (
                        list(reversed(remaining_indices))
                        if args.reverse_order
                        else remaining_indices
                    )
                    warmup_indices, warmup_policy = _select_warmup_indices(
                        count,
                        processing_indices,
                        args.warmup,
                        _sampling_seed(args.sampling_seed, task.name),
                    ) if processing_indices else ([], "skipped_no_new_queries")
                    measurement_seed = getattr(adapter, "seed", args.seed)
                    def run_sample(_index: int, selected_condition=condition):
                        return adapter.run(selected_condition)

                    current_latency = (
                        benchmark_dataset_callable(
                            run_sample,
                            sample_indices=processing_indices,
                            warmup_indices=warmup_indices,
                            seed=measurement_seed,
                            prepare_sample=lambda index: _select_and_prime_inputs(
                                adapter, index
                            ),
                        )
                        if processing_indices
                        else None
                    )
                    latency = _merge_latency_reports(
                        previous_latency,
                        current_latency,
                        target_indices,
                        resume_path,
                        "reverse" if args.reverse_order else "forward",
                    )
                    flop_sample_limit = (
                        args.flops_samples_per_task if condition == "ours" else 1
                    )
                    flop_sample_indices = _select_flop_indices(
                        target_indices, flop_sample_limit
                    )
                    adapter.select_sample(flop_sample_indices[0])
                    task_result: dict[str, Any] = {
                        "task": task.name,
                        "dataset_json": str(json_path),
                        "dataset_records": count,
                        "source_dataset_records": source_record_count,
                        "record_filter": {
                            "task_a": task.task_a,
                            "task_b": task.task_b,
                            "source_task_a": task.source_task_a,
                            "source_task_b": task.source_task_b,
                            "source_record_indices": record_indices,
                        },
                        "demo_input": task.demo_input,
                        "demo_output": task.demo_output,
                        "manifest_task_instruction": task.text_prompt,
                        "metadata": adapter.condition_metadata(condition),
                        "latency": latency,
                        "warmup_policy": warmup_policy,
                        "target_sample_indices": target_indices,
                        "representative_flops_sample_index": flop_sample_indices[0],
                        "flops_profile_sample_indices": flop_sample_indices,
                        "flops": None,
                    }
                    condition_result["tasks"].append(task_result)

                if args.profile_flops:
                    for task_result, task in zip(condition_result["tasks"], tasks):
                        print(f"FLOPs: condition={condition}, task={task.name}...")
                        adapter.configure_samples(
                            Path(task_result["dataset_json"]),
                            demo_input=task.demo_input,
                            demo_output=task.demo_output,
                            record_indices=task_result["record_filter"][
                                "source_record_indices"
                            ],
                        )
                        _set_task_prompt(adapter, task.text_prompt, task.name)
                        profiles = []
                        for sample_index in task_result[
                            "flops_profile_sample_indices"
                        ]:
                            adapter.select_sample(sample_index)
                            profiles.append(
                                profile_flops(
                                    lambda c=condition: adapter.run(c),
                                    components=components,
                                    top_k=args.flops_top_k,
                                    seed=getattr(adapter, "seed", args.seed),
                                )
                            )
                        task_result["flops"] = _aggregate_flop_profiles(
                            task_result["flops_profile_sample_indices"], profiles
                        )
                condition_result["aggregate"] = _aggregate_tasks(
                    condition_result["tasks"]
                )
                document["conditions"].append(condition_result)
            finally:
                components = None
                adapter.release_condition(condition)
    finally:
        adapter.close()
    return document


def _rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in document["conditions"]:
        params = condition["parameters"]["unique_total"]
        if not condition["tasks"]:
            rows.append(
                {
                    "adapter": document["adapter"],
                    "condition": condition["condition"],
                    "task": "",
                    "logical_parameters": params["total"],
                    "parameter_storage_bytes": params["storage_bytes"],
                    "trained_parameters": condition["trained_parameters"],
                }
            )
            continue
        for task in condition["tasks"]:
            latency = task["latency"]["end_to_end"]
            flops = task["flops"] or {}
            rows.append(
                {
                    "adapter": document["adapter"],
                    "condition": condition["condition"],
                    "task": task["task"],
                    "resolution": document["controlled_protocol"]["resolution"],
                    "measured_queries": latency["count"],
                    "latency_mean_s": latency["mean_s"],
                    "latency_median_s": latency["median_s"],
                    "latency_std_s": latency["std_s"],
                    "latency_p90_s": latency["p90_s"],
                    "latency_p95_s": latency["p95_s"],
                    "logical_parameters": params["total"],
                    "parameter_storage_bytes": params["storage_bytes"],
                    "trained_parameters": condition["trained_parameters"],
                    "estimated_total_flops": flops.get("estimated_total_flops"),
                    "flops_min": flops.get("estimated_total_flops_min"),
                    "flops_max": flops.get("estimated_total_flops_max"),
                    "flops_profile_samples": flops.get("profile_sample_count"),
                    "flops_status": flops.get("status"),
                    "flops_confidence": flops.get("confidence"),
                }
            )
        aggregate = condition.get("aggregate")
        if aggregate:
            rows.append(
                {
                    "adapter": document["adapter"],
                    "condition": condition["condition"],
                    "task": "__macro__",
                    "resolution": document["controlled_protocol"]["resolution"],
                    "measured_queries": aggregate["queries"],
                    "latency_mean_s": aggregate["macro_task_mean_latency_s"],
                    "latency_median_s": aggregate["macro_task_median_latency_s"],
                    "latency_std_s": None,
                    "latency_p90_s": None,
                    "latency_p95_s": None,
                    "logical_parameters": params["total"],
                    "parameter_storage_bytes": params["storage_bytes"],
                    "trained_parameters": condition["trained_parameters"],
                    "estimated_total_flops": aggregate["macro_task_flops"],
                    "flops_status": (
                        "macro_all_tasks_complete"
                        if aggregate["flops_complete_for_all_tasks"]
                        else "macro_unavailable_or_partial"
                    ),
                    "flops_confidence": None,
                }
            )
    return rows


def write_suite(document: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"{document['adapter']}_suite_{stamp}.json"
    latest_path = output_dir / f"{document['adapter']}_suite_latest.json"
    payload = json.dumps(document, indent=2)
    json_path.write_text(payload, encoding="utf-8")
    latest_path.write_text(payload, encoding="utf-8")

    rows = _rows(document)
    csv_path = output_dir / f"{document['adapter']}_suite.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return json_path


def parser():
    result = benchmark_parser()
    result.description = (
        "Measure parameters once and fixed-shape FLOPs/latency over distinct queries per task"
    )
    for action in result._actions:
        if action.dest == "adapter":
            action.choices = [
                choice for choice in action.choices if choice != "toy"
            ]
        elif action.dest in {
            "dataset_json",
            "demo_input",
            "demo_output",
            "repeats",
            "sample_index",
            "shape_policy",
        }:
            action.help = argparse.SUPPRESS
    result.add_argument(
        "--task-manifest",
        type=Path,
        default=None,
        help=(
            "Task manifest; defaults to t2t_tasks.json for t2t-* adapters "
            "and competitor_tasks.json for paper-facing official baselines"
        ),
    )
    result.add_argument("--tasks", nargs="+")
    result.add_argument("--max-samples", type=int, default=100)
    result.add_argument("--sampling-seed", type=int, default=2026)
    result.add_argument(
        "--resume-from",
        type=Path,
        help=(
            "Reuse raw latency rows from a smaller compatible suite JSON and measure "
            "only missing target indices"
        ),
    )
    result.add_argument(
        "--reverse-order",
        action="store_true",
        help="Measure missing query indices from highest to lowest",
    )
    result.add_argument(
        "--flops-samples-per-task",
        type=int,
        default=5,
        help=(
            "Measured queries used to average input-dependent Ours FLOPs; "
            "fixed/official conditions use one because their controlled shapes are fixed"
        ),
    )
    result.set_defaults(
        warmup=5,
        resolution=None,
        shape_policy="fixed-square",
        data_root=None,
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    document = run_suite(args)
    project_root = Path(__file__).resolve().parents[1]
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir.is_absolute()
        else (project_root / args.output_dir).resolve()
    )
    path = write_suite(document, output_dir)
    print(f"Wrote {path}")
    print(f"Wrote {output_dir / (document['adapter'] + '_suite.csv')}")


if __name__ == "__main__":
    main()
