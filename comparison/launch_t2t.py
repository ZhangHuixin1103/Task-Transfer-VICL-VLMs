"""Launch independent T2T-VICL comparison jobs with one visible GPU per model."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Sequence


ADAPTERS = ("t2t-qwen", "t2t-flux2", "t2t-omnigen2", "t2t-firered")


@dataclass(frozen=True)
class Job:
    adapter: str
    gpu: str
    python: str
    model_id: str


@dataclass
class RunningJob:
    job: Job
    command: list[str]
    log_path: Path
    log_handle: IO[str]
    process: subprocess.Popen[str]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--job",
        action="append",
        nargs=4,
        required=True,
        metavar=("ADAPTER", "GPU", "PYTHON", "MODEL_ID"),
        help=(
            "Repeat once per model, for example: --job t2t-qwen 1 "
            "/envs/qwen/bin/python /weights/Qwen-Image-Edit-2511"
        ),
    )
    result.add_argument(
        "--conditions",
        nargs="+",
        choices=("fixed", "ours"),
        default=("fixed", "ours"),
    )
    result.add_argument(
        "--prompt-checkpoint",
        help="Required when --conditions contains ours",
    )
    result.add_argument("--prompt-base-model", default="Qwen/Qwen3-VL-4B-Instruct")
    result.add_argument(
        "--task-manifest",
        type=Path,
        default=Path(__file__).with_name("t2t_tasks.json"),
    )
    result.add_argument("--tasks", nargs="+")
    result.add_argument("--max-samples", type=int, default=100)
    result.add_argument("--warmup", type=int, default=5)
    result.add_argument(
        "--resolution",
        type=int,
        help="Optional square-size override; omit it to keep each editor's default",
    )
    result.add_argument("--steps", type=int)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--sampling-seed", type=int, default=2026)
    result.add_argument("--flops-samples-per-task", type=int, default=5)
    result.add_argument("--profile-flops", action="store_true")
    result.add_argument("--parameters-only", action="store_true")
    result.add_argument("--firered-optimized", action="store_true")
    result.add_argument(
        "--cpu-threads-per-job",
        type=int,
        help="Set OMP_NUM_THREADS and MKL_NUM_THREADS in every child",
    )
    result.add_argument(
        "--output-root",
        type=Path,
        default=Path("comparison/outputs/resources/parallel"),
    )
    result.add_argument(
        "--fail-fast",
        action="store_true",
        help="Terminate remaining jobs after the first nonzero exit",
    )
    result.add_argument(
        "--dry-run", action="store_true", help="Print commands without launching them"
    )
    return result


def _resolve_python(reference: str) -> str:
    candidate = Path(reference).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    executable = shutil.which(reference)
    if executable:
        return executable
    raise FileNotFoundError(f"Python executable not found: {reference}")


def _parse_jobs(raw_jobs: list[list[str]]) -> list[Job]:
    jobs = []
    for adapter, gpu, python, model_id in raw_jobs:
        if adapter not in ADAPTERS:
            raise ValueError(f"Unsupported adapter {adapter!r}; choose from {ADAPTERS}")
        if not gpu or "," in gpu:
            raise ValueError(
                f"Each --job must receive exactly one GPU id or UUID, got {gpu!r}"
            )
        jobs.append(Job(adapter, gpu, _resolve_python(python), model_id))

    duplicate_adapters = {
        job.adapter
        for job in jobs
        if sum(other.adapter == job.adapter for other in jobs) > 1
    }
    duplicate_gpus = {
        job.gpu for job in jobs if sum(other.gpu == job.gpu for other in jobs) > 1
    }
    if duplicate_adapters:
        raise ValueError(f"Adapters must be unique: {sorted(duplicate_adapters)}")
    if duplicate_gpus:
        raise ValueError(f"One GPU cannot host two jobs: {sorted(duplicate_gpus)}")
    return jobs


def _job_command(args, job: Job, project_root: Path, output_root: Path) -> list[str]:
    command = [
        job.python,
        "-u",
        "-m",
        "comparison.suite",
        "--adapter",
        job.adapter,
        "--conditions",
        *args.conditions,
        "--task-manifest",
        str(args.task_manifest.expanduser().resolve()),
        "--model-id",
        job.model_id,
        "--prompt-base-model",
        args.prompt_base_model,
        "--max-samples",
        str(args.max_samples),
        "--warmup",
        str(args.warmup),
        "--seed",
        str(args.seed),
        "--sampling-seed",
        str(args.sampling_seed),
        "--flops-samples-per-task",
        str(args.flops_samples_per_task),
        "--output-dir",
        str(output_root / job.adapter),
    ]
    if args.resolution is not None:
        command.extend(("--resolution", str(args.resolution)))
    if args.prompt_checkpoint:
        command.extend(("--prompt-checkpoint", args.prompt_checkpoint))
    if args.tasks:
        command.extend(("--tasks", *args.tasks))
    if args.steps is not None:
        command.extend(("--steps", str(args.steps)))
    if args.profile_flops:
        command.append("--profile-flops")
    if args.parameters_only:
        command.append("--parameters-only")
    if args.firered_optimized and job.adapter == "t2t-firered":
        command.append("--optimized")
    return command


def _child_environment(args, job: Job) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = job.gpu
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    environment["PYTHONHASHSEED"] = str(args.sampling_seed)
    if args.cpu_threads_per_job is not None:
        threads = str(args.cpu_threads_per_job)
        environment["OMP_NUM_THREADS"] = threads
        environment["MKL_NUM_THREADS"] = threads
    return environment


def _terminate(running: list[RunningJob]) -> None:
    alive = [item for item in running if item.process.poll() is None]
    for item in alive:
        item.process.send_signal(signal.SIGTERM)
    deadline = time.monotonic() + 15
    while alive and time.monotonic() < deadline:
        alive = [item for item in alive if item.process.poll() is None]
        time.sleep(0.2)
    for item in alive:
        item.process.kill()


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if "ours" in args.conditions and not args.prompt_checkpoint:
        raise SystemExit("--prompt-checkpoint is required for the ours condition")
    if args.max_samples < 1 or args.warmup < 0:
        raise SystemExit("max samples must be positive and warmup nonnegative")
    if args.resolution is not None and args.resolution < 1:
        raise SystemExit("resolution must be positive")
    if args.cpu_threads_per_job is not None and args.cpu_threads_per_job < 1:
        raise SystemExit("--cpu-threads-per-job must be positive")

    project_root = Path(__file__).resolve().parents[1]
    jobs = _parse_jobs(args.job)
    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root = output_root.resolve()
    log_root = output_root / "logs"
    commands = [_job_command(args, job, project_root, output_root) for job in jobs]

    print(
        "Parallel CUDA isolation does not isolate shared CPU, RAM, storage, PCIe, or "
        "power resources. Confirm final latency against a one-job-at-a-time run."
    )
    for job, command in zip(jobs, commands):
        print(f"[{job.adapter} physical GPU {job.gpu}] {shlex.join(command)}")
    if args.dry_run:
        return

    log_root.mkdir(parents=True, exist_ok=True)
    running: list[RunningJob] = []
    started = datetime.now(timezone.utc).isoformat()
    try:
        for job, command in zip(jobs, commands):
            log_path = log_root / f"{job.adapter}.log"
            log_handle = log_path.open("w", encoding="utf-8", buffering=1)
            process = subprocess.Popen(
                command,
                cwd=project_root,
                env=_child_environment(args, job),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running.append(RunningJob(job, command, log_path, log_handle, process))
            print(f"Started {job.adapter}: pid={process.pid}, log={log_path}")

        failed = False
        pending = list(running)
        while pending:
            for item in list(pending):
                return_code = item.process.poll()
                if return_code is None:
                    continue
                pending.remove(item)
                state = "completed" if return_code == 0 else "failed"
                print(f"{item.job.adapter} {state} with exit code {return_code}")
                failed = failed or return_code != 0
                if return_code != 0 and args.fail_fast:
                    _terminate(pending)
                    pending.clear()
                    break
            if pending:
                time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted; terminating child jobs...")
        _terminate(running)
        failed = True
    finally:
        for item in running:
            item.log_handle.close()

    manifest = {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "parallel_jobs": [
            {
                "adapter": item.job.adapter,
                "physical_gpu": item.job.gpu,
                "python": item.job.python,
                "model_id": item.job.model_id,
                "pid": item.process.pid,
                "return_code": item.process.poll(),
                "log": str(item.log_path),
                "command": item.command,
            }
            for item in running
        ],
    }
    manifest_path = output_root / "launch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
