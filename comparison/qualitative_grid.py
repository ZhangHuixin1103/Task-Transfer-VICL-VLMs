"""Create an aligned appendix grid from saved comparison predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _method_label(document: dict[str, Any], condition: str) -> str:
    adapter = str(document["adapter"])
    labels = {
        "mae-vqgan": "MAE-VQGAN",
        "painter": "Painter",
        "prompt-diffusion": "PromptDiff.",
        "instruct-diffusion": "InstructDiff.",
        "visualcloze": "VisualCloze",
        "prompt-gip": "PromptGIP",
    }
    label = labels.get(adapter, adapter)
    return label if condition == "official" else f"{label}:{condition}"


def load_quality_predictions(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("kind") != "image_quality_suite":
        raise ValueError(f"Not an image_quality_suite result: {path}")
    output: dict[str, dict[str, dict[str, str]]] = {}
    for condition in document.get("conditions") or []:
        label = _method_label(document, str(condition.get("condition")))
        records: dict[str, dict[str, str]] = {}
        for task in condition.get("tasks") or []:
            records_path = Path(task["records_jsonl"])
            if not records_path.is_file():
                records_path = path.parent / "records" / records_path.name
            for row in _load_jsonl(records_path):
                prediction = row.get("output_path")
                sample = row.get("sample") or {}
                prediction_path = Path(prediction) if prediction else None
                if prediction_path is not None and not prediction_path.is_file():
                    prediction_path = (
                        path.parent
                        / "images"
                        / str(condition.get("condition"))
                        / str(task["task"])
                        / prediction_path.name
                    )
                if prediction_path is not None and prediction_path.is_file():
                    records[str(sample["task_b_input"])] = {
                        "prediction": str(prediction_path.resolve()),
                        "target": str(sample["task_b_output"]),
                        "demo": str(sample["task_a_input"]),
                        "task": str(task["task"]),
                    }
        output[label] = records
    return output


def _combo_id(task_a_input: str, task_b_input: str) -> str:
    digest = hashlib.sha1()
    for value in (task_a_input, task_b_input):
        digest.update(value.encode("utf-8"))
        digest.update(b"|")
    return digest.hexdigest()[:10]


def load_legacy_predictions(
    specification: str, eval_json: Path
) -> tuple[str, dict[str, dict[str, str]]]:
    try:
        method, raw_root = specification.split("=", 1)
    except ValueError as error:
        raise ValueError("--legacy must have the form METHOD=OUTPUT_ROOT") from error
    root = Path(raw_root).expanduser().resolve()
    rows = json.loads(eval_json.read_text(encoding="utf-8"))
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        task_a = str(row["taskA_input"]).split("/", 1)[0]
        task_b = str(row["taskB_input"]).split("/", 1)[0]
        pair = f"{task_a}__{task_b}"
        log_path = root / pair / "evaluation_log.jsonl"
        if not log_path.is_file():
            continue
        logged = {entry["combo_id"]: entry for entry in _load_jsonl(log_path)}
        entry = logged.get(_combo_id(row["taskA_input"], row["taskB_input"]))
        if entry is None:
            continue
        prediction = root / pair / Path(entry["final_image"]).name
        if not prediction.is_file():
            continue
        output.setdefault(
            str(row["taskB_input"]),
            {
                "prediction": str(prediction),
                "target": str(row["taskB_output"]),
                "task": task_b,
            },
        )
    return method, output


def load_rebuttal_first_predictions(
    specification: str,
) -> tuple[str, dict[str, dict[str, str]]]:
    try:
        method, raw_root = specification.split("=", 1)
    except ValueError as error:
        raise ValueError(
            "--rebuttal-first must have the form METHOD=OUTPUT_ROOT"
        ) from error
    root = Path(raw_root).expanduser().resolve()
    log_path = root / "evaluation_log.jsonl"
    output: dict[str, dict[str, str]] = {}
    for row in _load_jsonl(log_path):
        first_image = row.get("first_image")
        if not first_image:
            continue
        prediction = (
            root
            / str(row["pair_key"])
            / str(row["combo_id"])
            / Path(first_image).name
        )
        if not prediction.is_file():
            continue
        task_b_input = str(row["taskB_input"])
        output[task_b_input] = {
            "prediction": str(prediction),
            "target": str(row["taskB_output"]),
            "task": task_b_input.split("/", 1)[0],
        }
    if not output:
        raise ValueError(f"No saved first-shot predictions found below {root}")
    return method, output


def _cell(path: Path, size: int, label: str) -> Image.Image:
    band = 32
    canvas = Image.new("RGB", (size, size + band), "white")
    with Image.open(path) as source:
        image = ImageOps.contain(source.convert("RGB"), (size, size))
        canvas.paste(image, ((size - image.width) // 2, band + (size - image.height) // 2))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.text((6, 6), label, fill="black", font=font)
    return canvas


def build_grid(
    methods: dict[str, dict[str, dict[str, str]]],
    data_root: Path,
    output: Path,
    tasks: Sequence[str] | None,
    samples_per_task: int,
    cell_size: int,
    seed: int,
) -> None:
    if not methods:
        raise ValueError("No prediction sources supplied")
    common = set.intersection(*(set(records) for records in methods.values()))
    first_records = next(iter(methods.values()))
    available_tasks = sorted({first_records[key]["task"] for key in common})
    selected_tasks = list(tasks or available_tasks)
    unknown = set(selected_tasks) - set(available_tasks)
    if unknown:
        raise ValueError(f"No common saved predictions for tasks: {sorted(unknown)}")

    selected: list[tuple[str, str]] = []
    for task in selected_tasks:
        candidates = sorted(key for key in common if first_records[key]["task"] == task)
        if len(candidates) < samples_per_task:
            raise ValueError(f"{task} has only {len(candidates)} common predictions")
        task_rng = random.Random(f"{seed}:{task}")
        for key in sorted(task_rng.sample(candidates, samples_per_task)):
            selected.append((task, key))

    labels = ["Query", "Ground truth", *methods]
    grid = Image.new(
        "RGB",
        (cell_size * len(labels), (cell_size + 32) * len(selected)),
        "white",
    )
    for row_index, (task, key) in enumerate(selected):
        record = first_records[key]
        paths = [
            data_root / key,
            data_root / record["target"],
            *(Path(methods[label][key]["prediction"]) for label in methods),
        ]
        for column, (label, path) in enumerate(zip(labels, paths)):
            row_label = task.replace("_", " ").title()
            rendered = _cell(path, cell_size, f"{row_label}: {label}" if column == 0 else label)
            grid.paste(rendered, (column * cell_size, row_index * (cell_size + 32)))
            rendered.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output)
    grid.close()


def parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("results", nargs="*", type=Path)
    result.add_argument("--legacy", action="append", default=[])
    result.add_argument("--rebuttal-first", action="append", default=[])
    result.add_argument(
        "--eval-json", type=Path, default=project_root / "data/dataset/eval_dataset.json"
    )
    result.add_argument("--data-root", type=Path, default=project_root / "data/tasks")
    result.add_argument("--tasks", nargs="+")
    result.add_argument("--samples-per-task", type=int, default=1)
    result.add_argument("--cell-size", type=int, default=224)
    result.add_argument("--seed", type=int, default=2026)
    result.add_argument(
        "--output",
        type=Path,
        default=project_root / "comparison/outputs/paper_tables/qualitative_grid.png",
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    methods: dict[str, dict[str, dict[str, str]]] = {}
    for path in args.results:
        for label, records in load_quality_predictions(path.expanduser().resolve()).items():
            if label in methods:
                raise ValueError(f"Duplicate method label: {label}")
            methods[label] = records
    for specification in args.legacy:
        label, records = load_legacy_predictions(
            specification, args.eval_json.expanduser().resolve()
        )
        if label in methods:
            raise ValueError(f"Duplicate method label: {label}")
        methods[label] = records
    for specification in args.rebuttal_first:
        label, records = load_rebuttal_first_predictions(specification)
        if label in methods:
            raise ValueError(f"Duplicate method label: {label}")
        methods[label] = records
    build_grid(
        methods,
        args.data_root.expanduser().resolve(),
        args.output.expanduser().resolve(),
        args.tasks,
        args.samples_per_task,
        args.cell_size,
        args.seed,
    )
    print(f"Wrote {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
