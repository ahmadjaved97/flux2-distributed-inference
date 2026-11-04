"""
Generate food images from a CSV using FLUX.2-dev with the distributed Flux2 pipeline.

Backend:
  - Uses YigitEkin/flux2-distributed-inference
  - Loads black-forest-labs/FLUX.2-dev through Flux2Pipeline.from_pretrained_distributed(...)
  - Uses Accelerate device_map="auto" model-parallel sharding across visible GPUs.

Input CSV columns required:
  image_path,dataset_name,class_id,is_human,description,gen_model

Key behavior preserved from the Klein CSV script:
  - image_path values are relative to --input_root.
  - generation prompt uses the description column by default; override with --prompt_column.
  - output CSV keeps the original columns and adds only:
      generated_image_path, status
  - errors are recorded in the output CSV only as status="error".
  - detailed error/seed/timing info is written to JSON, not the CSV.
  - per-item JSON entries do not store prompts; reproducibility settings are stored once.
  - default run output folder is <output_folder>/<model>_<dataset>[_append_word].
  - default CSV and JSON logs are saved directly inside that run output folder.
  - output CSV is append-only and resume is based on existing successful rows.

Example:
python flux2_dev_image_generation.py \
  --input input.csv \
  --input_root /data/original_images \
  --output_folder /data/generated_runs \
  --model FLUX2-dev \
  --seed 42 \
  --height 1024 \
  --width 1024
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing as mp
import platform
import queue as queue_mod
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch

try:
    # Requires the YigitEkin/flux2-distributed-inference repository to be installed
    # or available on PYTHONPATH, e.g.:
    #   git clone https://github.com/YigitEkin/flux2-distributed-inference.git
    #   export PYTHONPATH=/path/to/flux2-distributed-inference:$PYTHONPATH
    from src.flux2_pipeline import Flux2Pipeline
except ImportError as exc:
    raise ImportError(
        "Could not import Flux2Pipeline from YigitEkin/flux2-distributed-inference. "
        "Clone/install the repository and add it to PYTHONPATH."
    ) from exc


MODEL_REPOS: Dict[str, str] = {
    "FLUX2-dev": "black-forest-labs/FLUX.2-dev",
}

BASE_REQUIRED_COLUMNS = ["image_path", "dataset_name", "class_id", "is_human", "gen_model"]
ADDED_OUTPUT_COLUMNS = ["generated_image_path", "status"]
SUCCESS_STATUSES = {"generated", "reused"}

SYSTEM_PROMPT = ""
BASE_PROMPT_TEMPLATE = (
    "Generate a high-quality, realistic food photograph based only on this food description. "
    "Match the food appearance, ingredients, serving vessel, scene composition, lighting, "
    "camera angle, and image quality details as closely as possible.\n\n"
    "{prompt_text}"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_name(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "model"


def infer_dataset_name(rows: List[Dict[str, str]], input_csv: Path) -> str:
    dataset_names = sorted({(row.get("dataset_name") or "").strip() for row in rows if (row.get("dataset_name") or "").strip()})
    if len(dataset_names) == 1:
        return sanitize_name(dataset_names[0])
    if len(dataset_names) > 1:
        preview = "_".join(sanitize_name(name) for name in dataset_names[:3])
        if len(dataset_names) > 3:
            return f"{preview}_mixed_{len(dataset_names)}_datasets"
        return preview
    return sanitize_name(input_csv.stem)


def build_run_output_folder(base_output_folder: Path, model_name: str, dataset_name: str, output_folder_suffix: str) -> Path:
    parts = [sanitize_name(model_name), sanitize_name(dataset_name)]
    suffix = sanitize_name(output_folder_suffix) if output_folder_suffix.strip() else ""
    if suffix:
        parts.append(suffix)
    return base_output_folder / "_".join(parts)


def resolve_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = dtype_str.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_str}'. Choose from: {sorted(mapping)}")
    return mapping[key]


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("flux2_dev_csv_gen")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def worker_logger(worker_id: int) -> logging.Logger:
    logger = logging.getLogger(f"flux_worker_{worker_id}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(f"%(asctime)s [worker {worker_id}] [%(levelname)s] %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def load_input_rows(input_csv: Path, prompt_column: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        required_columns = list(BASE_REQUIRED_COLUMNS)
        if prompt_column not in required_columns:
            required_columns.append(prompt_column)
        missing = [c for c in required_columns if c not in fieldnames]
        if missing:
            raise ValueError(f"Input CSV missing required columns: {missing}. Found: {fieldnames}")
        return [dict(r) for r in reader], fieldnames


def output_fieldnames(input_fieldnames: Iterable[str]) -> List[str]:
    fields = list(input_fieldnames)
    for col in ADDED_OUTPUT_COLUMNS:
        if col not in fields:
            fields.append(col)
    return fields


def read_completed_keys(output_csv: Path, model_name: str) -> Set[Tuple[str, str]]:
    completed: Set[Tuple[str, str]] = set()
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return completed
    with output_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = (row.get("image_path") or "").strip()
            status = (row.get("status") or "").strip().lower()
            generated_image_path = (row.get("generated_image_path") or "").strip()

            # Backward compatibility: older output CSVs may contain generated_model.
            # New output folders are model-specific, so new CSVs do not need this column.
            old_generated_model = (row.get("generated_model") or "").strip()
            model_matches = (not old_generated_model) or (old_generated_model == model_name)

            if image_path and model_matches and status in SUCCESS_STATUSES and generated_image_path:
                completed.add((image_path, model_name))
    return completed


def append_output_row(output_csv: Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists() and output_csv.stat().st_size > 0
    with output_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def build_prompt(prompt_text: str) -> str:
    return BASE_PROMPT_TEMPLATE.format(prompt_text=prompt_text.strip())


def build_output_image_path(image_path: str, output_folder: Path, model_name: str, image_extension: str) -> Path:
    rel = Path(image_path)
    suffix = image_extension if image_extension.startswith(".") else f".{image_extension}"
    return output_folder / rel.parent / f"{rel.stem}_{sanitize_name(model_name)}{suffix}"


def parse_max_memory(max_memory_str: str) -> Optional[Dict[Any, str]]:
    """Parse a simple max-memory string for Accelerate device maps.

    Format examples:
      "0:70GB,1:70GB,2:70GB,3:70GB,cpu:100GB"
      "70GB"  -> applies to every visible GPU and adds cpu:100GB
    """
    value = (max_memory_str or "").strip()
    if not value:
        return None

    if ":" not in value:
        memory: Dict[Any, str] = {i: value for i in range(torch.cuda.device_count())}
        memory["cpu"] = "100GB"
        return memory

    memory = {}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        key, mem = part.split(":", 1)
        key = key.strip()
        mem = mem.strip()
        memory["cpu" if key.lower() == "cpu" else int(key)] = mem
    return memory or None


def load_pipeline(repo_id: str, dtype: torch.dtype, device: str, load_strategy: str, cpu_offload: bool, logger: logging.Logger, max_memory: Optional[Dict[Any, str]]) -> Flux2Pipeline:
    logger.info("Loading FLUX.2-dev distributed pipeline from %s", repo_id)
    if not torch.cuda.is_available():
        raise RuntimeError("FLUX.2-dev distributed inference requires CUDA GPUs.")
    if cpu_offload:
        logger.warning("--cpu_offload is kept for CLI compatibility but is not used with from_pretrained_distributed().")
    if load_strategy != "auto":
        logger.warning("--load_strategy=%s is kept for CLI compatibility; FLUX.2-dev uses from_pretrained_distributed().", load_strategy)

    pipe = Flux2Pipeline.from_pretrained_distributed(
        repo_id,
        torch_dtype=dtype,
        max_memory=max_memory,
        text_encoder_device_map="auto",
        transformer_device_map="auto",
        vae_device_map=None,
    )
    if hasattr(pipe, "get_device_distribution"):
        logger.info("Device distribution: %s", pipe.get_device_distribution())
    return pipe

def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpus"] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["gpus"].append({"index": i, "name": torch.cuda.get_device_name(i), "total_memory_gb": round(props.total_memory / (1024**3), 2)})
    return info


def make_error_result(row_index: int, row: Dict[str, str], status_error: str, output_image_path: Optional[Path], args_dict: Dict[str, Any], item_started: float, item_started_at: str) -> Dict[str, Any]:
    image_path = (row.get("image_path") or "").strip()
    prompt_column = args_dict["prompt_column"]
    prompt_text = (row.get(prompt_column) or "").strip()
    return {
        "row_index": row_index,
        "row": row,
        "status": "error",
        "generated_image_path": "",
        "generated_model": args_dict["model"],
        "hf_model_repo_id": args_dict["repo_id"],
        "item_log": {
            "row_index": row_index,
            "source_image_path": image_path,
            "source_image_abs_path": str(Path(args_dict["input_root"]) / image_path) if image_path else "",
            "input_gen_model_value": row.get("gen_model", ""),
            "generated_model": args_dict["model"],
            "hf_model_repo_id": args_dict["repo_id"],
            "generated_image_path": str(output_image_path) if output_image_path else "",
            "prompt_column": prompt_column,
            "seed": args_dict["seed"],
            "started_at": item_started_at,
            "finished_at": utc_now_iso(),
            "time_seconds": round(time.perf_counter() - item_started, 4),
            "status": "error",
                "error": status_error,
        },
    }


def worker(worker_id: int, device: str, indexed_rows: List[Tuple[int, Dict[str, str]]], args_dict: Dict[str, Any], result_queue: "mp.Queue[Dict[str, Any]]") -> None:
    logger = worker_logger(worker_id)
    dtype = resolve_dtype(args_dict["dtype"])
    try:
        pipe = load_pipeline(args_dict["repo_id"], dtype, device, args_dict["load_strategy"], args_dict["cpu_offload"], logger, args_dict["max_memory"])
    except Exception as exc:
        for row_index, row in indexed_rows:
            now_perf = time.perf_counter()
            now_iso = utc_now_iso()
            result_queue.put(make_error_result(row_index, row, f"pipeline_load_error: {exc}", None, args_dict, now_perf, now_iso))
        return

    input_root = Path(args_dict["input_root"])
    output_folder = Path(args_dict["output_folder"])

    for row_index, row in indexed_rows:
        item_started = time.perf_counter()
        item_started_at = utc_now_iso()
        image_path = (row.get("image_path") or "").strip()
        prompt_column = args_dict["prompt_column"]
        prompt_text = (row.get(prompt_column) or "").strip()
        output_image_path = build_output_image_path(image_path, output_folder, args_dict["model"], args_dict["image_extension"])

        if not image_path:
            result_queue.put(make_error_result(row_index, row, "empty_image_path", output_image_path, args_dict, item_started, item_started_at))
            continue
        if not prompt_text or prompt_text.upper() == "ERROR":
            result_queue.put(make_error_result(row_index, row, f"empty_or_error_prompt_column: {prompt_column}", output_image_path, args_dict, item_started, item_started_at))
            continue

        source_abs_path = input_root / image_path
        if not source_abs_path.exists():
            result_queue.put(make_error_result(row_index, row, f"source_image_not_found: {source_abs_path}", output_image_path, args_dict, item_started, item_started_at))
            continue

        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        prompt = build_prompt(prompt_text)

        if output_image_path.exists() and not args_dict["overwrite"]:
            result_queue.put({
                "row_index": row_index,
                "row": row,
                "status": "reused",
                "generated_image_path": str(output_image_path),
                "generated_model": args_dict["model"],
                "hf_model_repo_id": args_dict["repo_id"],
                "item_log": {
                    "row_index": row_index,
                    "source_image_path": image_path,
                    "source_image_abs_path": str(source_abs_path),
                    "input_gen_model_value": row.get("gen_model", ""),
                    "generated_model": args_dict["model"],
                    "hf_model_repo_id": args_dict["repo_id"],
                    "generated_image_path": str(output_image_path),
                    "prompt_column": prompt_column,
                    "seed": args_dict["seed"],
                    "started_at": item_started_at,
                    "finished_at": utc_now_iso(),
                    "time_seconds": round(time.perf_counter() - item_started, 4),
                    "status": "reused",
                    "error": None,
                },
            })
            continue

        try:
            generator_device = "cuda:0" if torch.cuda.is_available() else device
            generator = torch.Generator(device=generator_device).manual_seed(args_dict["seed"])
            gen_kwargs: Dict[str, Any] = {
                "prompt": prompt,
                "generator": generator,
                "num_inference_steps": args_dict["num_inference_steps"],
                "guidance_scale": args_dict["guidance_scale"],
            }
            if args_dict["height"] is not None:
                gen_kwargs["height"] = args_dict["height"]
            if args_dict["width"] is not None:
                gen_kwargs["width"] = args_dict["width"]

            result = pipe(**gen_kwargs)
            image = result.images[0]
            if output_image_path.suffix.lower() in {".jpg", ".jpeg"}:
                image = image.convert("RGB")
                image.save(output_image_path, quality=95)
            else:
                image.save(output_image_path)
            result_queue.put({
                "row_index": row_index,
                "row": row,
                "status": "generated",
                "generated_image_path": str(output_image_path),
                "generated_model": args_dict["model"],
                "hf_model_repo_id": args_dict["repo_id"],
                "item_log": {
                    "row_index": row_index,
                    "source_image_path": image_path,
                    "source_image_abs_path": str(source_abs_path),
                    "input_gen_model_value": row.get("gen_model", ""),
                    "generated_model": args_dict["model"],
                    "hf_model_repo_id": args_dict["repo_id"],
                    "generated_image_path": str(output_image_path),
                    "prompt_column": prompt_column,
                    "seed": args_dict["seed"],
                    "started_at": item_started_at,
                    "finished_at": utc_now_iso(),
                    "time_seconds": round(time.perf_counter() - item_started, 4),
                    "status": "generated",
                    "error": None,
                },
            })
        except Exception as exc:
            result_queue.put(make_error_result(row_index, row, str(exc), output_image_path, args_dict, item_started, item_started_at))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate images from a CSV using FLUX.2-dev with YigitEkin/flux2-distributed-inference.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--input_root", required=True, help="Root folder for the relative image_path values in the input CSV.")
    parser.add_argument("--output_folder", default="generated_outputs", help="Base folder for generated run folders. Default run folder: <output_folder>/<model>_<dataset>[_output_folder_suffix].")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REPOS), help="Model to use for every generated image.")
    parser.add_argument("--repo_id", default=None, help="Optional Hugging Face repo/local path override. If omitted, selected from --model.")
    parser.add_argument("--output_csv", default=None, help="Default: <run_output_folder>/generated_images.csv")
    parser.add_argument("--json_log", default=None, help="Default: <run_output_folder>/generation_metadata.json")
    parser.add_argument("--text_log", default=None, help="Default: <output_folder>/generation.log, shared/appended across runs")
    parser.add_argument("--output_folder_suffix", default="", help="Optional word/tag appended to the auto-created run output folder name.")
    parser.add_argument("--prompt_column", default="description", help="CSV column to use as the image-generation description/prompt text.")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42, help="Fixed seed used for every image.")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"])
    parser.add_argument("--max_memory", default="", help="Optional Accelerate max_memory for distributed FLUX.2-dev, e.g. '0:70GB,1:70GB,2:70GB,3:70GB,cpu:100GB' or '70GB'.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs. 0 means all visible GPUs. Uses CPU if CUDA is unavailable.")
    parser.add_argument("--cpu_offload", action="store_true", help="Enable diffusers CPU offload.")
    parser.add_argument("--load_strategy", choices=["auto", "full", "components"], default="auto")
    parser.add_argument("--image_extension", default=".jpg")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if the output image already exists.")
    parser.add_argument("--save_json_every", type=int, default=10, help="Update JSON log every N collected results.")
    args = parser.parse_args()

    run_start = time.perf_counter()
    input_csv = Path(args.input)
    input_root = Path(args.input_root)
    base_output_folder = Path(args.output_folder)
    repo_id = args.repo_id or MODEL_REPOS[args.model]
    rows, input_fieldnames = load_input_rows(input_csv, args.prompt_column)
    dataset_name = infer_dataset_name(rows, input_csv)
    output_folder = build_run_output_folder(base_output_folder, args.model, dataset_name, args.output_folder_suffix)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_csv = Path(args.output_csv) if args.output_csv else output_folder / "generated_images.csv"
    json_log = Path(args.json_log) if args.json_log else output_folder / "generation_metadata.json"
    text_log = Path(args.text_log) if args.text_log else base_output_folder / "generation.log"
    logger = setup_logger(text_log)

    fields = output_fieldnames(input_fieldnames)
    completed_keys = set() if args.overwrite else read_completed_keys(output_csv, args.model)

    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if cuda_count == 0:
        devices = ["cpu"]
    else:
        # The distributed FLUX.2-dev repository uses model-parallel sharding across
        # visible GPUs inside one process. Keep one worker to avoid loading one
        # sharded FLUX.2-dev copy per GPU/process.
        devices = ["cuda:0"]

    pending = [(idx, row) for idx, row in enumerate(rows) if (((row.get("image_path") or "").strip(), args.model) not in completed_keys)]

    logger.info("=" * 80)
    logger.info("FLUX.2-dev distributed CSV image generation")
    logger.info("Input CSV     : %s", input_csv)
    logger.info("Input root    : %s", input_root)
    logger.info("Base output   : %s", base_output_folder)
    logger.info("Dataset name  : %s", dataset_name)
    logger.info("Run output    : %s", output_folder)
    logger.info("Folder suffix : %s", args.output_folder_suffix or "<empty>")
    logger.info("Output CSV    : %s", output_csv)
    logger.info("JSON log      : %s", json_log)
    logger.info("Text log      : %s", text_log)
    logger.info("Model         : %s", args.model)
    logger.info("Repo ID       : %s", repo_id)
    logger.info("Prompt column  : %s", args.prompt_column)
    logger.info("Seed          : %s", args.seed)
    logger.info("Devices       : %s", devices)
    logger.info("Input rows    : %d", len(rows))
    logger.info("Completed     : %d", len(completed_keys))
    logger.info("Pending       : %d", len(pending))
    logger.info("=" * 80)

    run_data: Dict[str, Any] = {
        "run_started_at": utc_now_iso(),
        "run_finished_at": None,
        "total_time_seconds": None,
        "input_csv": str(input_csv),
        "input_root": str(input_root),
        "base_output_folder": str(base_output_folder),
        "dataset_name": dataset_name,
        "output_folder_suffix": args.output_folder_suffix,
        "output_folder": str(output_folder),
        "output_csv": str(output_csv),
        "json_log": str(json_log),
        "text_log": str(text_log),
        "model": args.model,
        "repo_id": repo_id,
        "python_version": platform.python_version(),
        "settings": {
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "seed_mode": "global_fixed_same_seed_for_every_row",
            "dtype": args.dtype,
            "max_memory": args.max_memory,
            "devices": devices,
            "cpu_offload": args.cpu_offload,
            "load_strategy": args.load_strategy,
            "image_extension": args.image_extension,
            "overwrite": args.overwrite,
            "prompt_column": args.prompt_column,
            "output_folder_suffix": args.output_folder_suffix,
        },
        "environment": environment_info(),
        "progress": {
            "total_input_rows": len(rows),
            "skipped_already_completed": len(completed_keys),
            "pending_this_run": len(pending),
            "processed_this_run": 0,
            "generated_this_run": 0,
            "reused_existing_image_this_run": 0,
            "error_this_run": 0,
        },
        "items": [],
        "reproducibility": {
            "prompt_column": args.prompt_column,
            "dataset_name": dataset_name,
            "output_folder_suffix": args.output_folder_suffix,
            "system_prompt": SYSTEM_PROMPT,
            "base_prompt_template": BASE_PROMPT_TEMPLATE,
            "prompt_construction": "BASE_PROMPT_TEMPLATE.format(prompt_text=<value from prompt_column>)",
            "per_item_prompts_stored": False,
            "model": args.model,
            "repo_id": repo_id,
            "python_version": platform.python_version(),
            "settings": {
                "height": args.height,
                "width": args.width,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
                "seed_mode": "global_fixed_same_seed_for_every_row",
                "dtype": args.dtype,
                "max_memory": args.max_memory,
                "cpu_offload": args.cpu_offload,
                "load_strategy": args.load_strategy,
                "image_extension": args.image_extension,
            },
        },
    }
    atomic_write_json(json_log, run_data)

    if not pending:
        logger.info("No pending rows. Nothing to do.")
        run_data["run_finished_at"] = utc_now_iso()
        run_data["total_time_seconds"] = round(time.perf_counter() - run_start, 4)
        atomic_write_json(json_log, run_data)
        return

    partitions: List[List[Tuple[int, Dict[str, str]]]] = [[] for _ in devices]
    for i, item in enumerate(pending):
        partitions[i % len(devices)].append(item)

    args_dict = {
        "input_root": str(input_root),
        "output_folder": str(output_folder),
        "dataset_name": dataset_name,
        "output_folder_suffix": args.output_folder_suffix,
        "model": args.model,
        "repo_id": repo_id,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "dtype": args.dtype,
        "max_memory": parse_max_memory(args.max_memory),
        "cpu_offload": args.cpu_offload,
        "load_strategy": args.load_strategy,
        "image_extension": args.image_extension,
        "overwrite": args.overwrite,
        "prompt_column": args.prompt_column,
    }

    mp_ctx = mp.get_context("spawn")
    result_queue: "mp.Queue[Dict[str, Any]]" = mp_ctx.Queue()
    processes: List[mp.Process] = []

    if len(devices) == 1:
        # In-process mode is easier to debug and avoids multiprocessing overhead.
        worker(0, devices[0], partitions[0], args_dict, result_queue)
    else:
        for worker_id, device in enumerate(devices):
            p = mp_ctx.Process(target=worker, args=(worker_id, device, partitions[worker_id], args_dict, result_queue), name=f"flux2-dev-worker-{worker_id}")
            p.start()
            processes.append(p)
            logger.info("Started worker %d on %s with %d row(s).", worker_id, device, len(partitions[worker_id]))

    collected = 0
    since_json_save = 0
    while collected < len(pending):
        try:
            res = result_queue.get(timeout=600)
        except queue_mod.Empty:
            if processes and all(not p.is_alive() for p in processes):
                logger.error("All workers stopped before all rows were returned.")
                break
            continue

        row = dict(res["row"])
        row.update({
            "generated_image_path": res.get("generated_image_path", ""),
            "status": res.get("status", "error"),
        })
        append_output_row(output_csv, fields, row)

        item_log = res["item_log"]
        run_data["items"].append(item_log)
        run_data["progress"]["processed_this_run"] += 1
        if res["status"] == "generated":
            run_data["progress"]["generated_this_run"] += 1
            logger.info("[%d/%d] generated: %s", collected + 1, len(pending), row.get("image_path", ""))
        elif res["status"] == "reused":
            run_data["progress"]["reused_existing_image_this_run"] += 1
            logger.info("[%d/%d] reused: %s", collected + 1, len(pending), row.get("image_path", ""))
        else:
            run_data["progress"]["error_this_run"] += 1
            logger.warning("[%d/%d] error: %s | %s", collected + 1, len(pending), row.get("image_path", ""), item_log.get("error"))

        collected += 1
        since_json_save += 1
        if since_json_save >= args.save_json_every:
            atomic_write_json(json_log, run_data)
            since_json_save = 0

    for p in processes:
        p.join(timeout=60)

    run_data["run_finished_at"] = utc_now_iso()
    run_data["total_time_seconds"] = round(time.perf_counter() - run_start, 4)
    atomic_write_json(json_log, run_data)

    logger.info("=" * 80)
    logger.info("Run complete")
    logger.info("Generated this run: %d", run_data["progress"]["generated_this_run"])
    logger.info("Reused this run   : %d", run_data["progress"]["reused_existing_image_this_run"])
    logger.info("Errors this run   : %d", run_data["progress"]["error_this_run"])
    logger.info("Output CSV        : %s", output_csv.resolve())
    logger.info("JSON log          : %s", json_log.resolve())
    logger.info("Text log          : %s", text_log.resolve())
    logger.info("=" * 80)


if __name__ == "__main__":
    main()