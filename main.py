import argparse
import asyncio
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.agents import SinglePromptGECAgent
from src.llm import create_router
from src.utils import process_in_parallel

load_dotenv()

MODE_LIMITS = {
    "fast": 10,
    "mini-test": 99,
    "full": None,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_config_path(config_path: str) -> Path:
    path = Path(config_path)
    candidates = [path] if path.is_absolute() else [
        Path.cwd() / path,
        repo_root() / path,
        Path(__file__).parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Configuration file not found. Checked: {checked}")


def load_config(config_path: str) -> tuple[dict, Path]:
    config_file = resolve_config_path(config_path)
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f), config_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ukrainian GEC evaluation.")
    parser.add_argument("--config", default="config.validation.yaml")
    return parser.parse_args()


def cleanup_litellm_async_clients() -> None:
    try:
        import litellm
    except Exception:
        return

    close_fn = getattr(litellm, "close_litellm_async_clients", None)
    if not callable(close_fn):
        return

    try:
        asyncio.run(close_fn())
    except Exception:
        pass


def marker_doc_id(text: str) -> str | None:
    match = re.fullmatch(r"\s*#\s*(\d+)\s*", text or "")
    return match.group(1) if match else None


def apply_mode_slice(
    sources: list[str],
    targets: list[str],
    start_index: int,
    limit: int | None,
) -> tuple[list[str], list[str]]:
    end_index = start_index + limit if limit is not None else None
    return sources[start_index:end_index], targets[start_index:end_index]


def load_dataset(config: dict) -> tuple[list[str], list[list[str]], list[dict], Path]:
    mode = config.get("mode", "fast")
    if mode not in MODE_LIMITS:
        raise ValueError(f"mode must be one of {list(MODE_LIMITS)}, got {mode!r}")

    partition = config.get("ua_partition", "valid")
    data_dir = config.get("ua_validation_dir")
    if not data_dir:
        raise ValueError("ua_validation_dir is required.")

    data_dir_path = Path(data_dir)
    if not data_dir_path.is_absolute():
        data_dir_path = repo_root() / data_dir_path
    src_path = data_dir_path / f"{partition}.src.txt"
    tgt_path = data_dir_path / f"{partition}.tgt.txt"
    m2_path = Path(config["ua_m2_path"]) if config.get("ua_m2_path") else data_dir_path / f"{partition}.m2"

    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    if not tgt_path.exists():
        raise FileNotFoundError(f"Target file not found: {tgt_path}")

    with open(src_path, "r", encoding="utf-8") as f:
        sources = [line.rstrip("\n") for line in f]
    with open(tgt_path, "r", encoding="utf-8") as f:
        targets = [line.rstrip("\n") for line in f]

    if len(sources) != len(targets):
        raise ValueError(f"Source/target length mismatch: {len(sources)} vs {len(targets)}")

    limit = config.get("limit", MODE_LIMITS[mode])
    start_index = int(config.get("start_index", 0))
    sources, targets = apply_mode_slice(sources, targets, start_index, limit)

    references = [[target] for target in targets]
    meta = []
    current_doc_id = "ua"
    for offset, source in enumerate(sources, start=start_index + 1):
        doc_id = marker_doc_id(source)
        if doc_id is not None:
            current_doc_id = doc_id
        meta.append(
            {
                "doc_id": current_doc_id,
                "sent_id": offset,
                "source_tokenized": source,
                "references_tokenized": [targets[offset - start_index - 1]],
            }
        )

    return sources, references, meta, m2_path


def build_router(config: dict):
    if config.get("llm_backend", "litellm_router") != "litellm_router":
        raise ValueError("Only llm_backend='litellm_router' is supported.")

    if config.get("llm_router"):
        return create_router(config["llm_router"])

    from src.llm.litellm import router

    return router


def validate_model(router, model: str) -> None:
    model_groups = sorted(
        {item["model_name"] for item in router.model_list if "model_name" in item}
    )
    if model not in model_groups:
        raise ValueError(
            f"Config model must match one of the router models: "
            f"{', '.join(model_groups)}. Got: {model!r}."
        )


def build_agent(config: dict) -> SinglePromptGECAgent:
    if config.get("agent", "gec_single_prompt") != "gec_single_prompt":
        raise ValueError("Only agent='gec_single_prompt' is supported.")

    model = config.get("model", "gpt-4.1-mini")
    router = build_router(config)
    validate_model(router, model)
    return SinglePromptGECAgent(
        model=model,
        temperature=config.get("temperature"),
        top_p=config.get("top_p"),
        request_timeout=float(config.get("llm_timeout_seconds", 120.0)),
        prompt_name=config.get("prompt_name", "base_en"),
        llm_router=router,
        reasoning_effort=config.get("reasoning_effort"),
    )


def passthrough_marker(source: str) -> str | None:
    doc_id = marker_doc_id(source)
    return f"# {doc_id}" if doc_id is not None else None


def process_sample(
    index: int,
    source: str,
    references: list[str],
    meta: dict,
    agent: SinglePromptGECAgent,
    log_input_output: bool,
) -> dict:
    corrected = passthrough_marker(source)
    error = None

    if corrected is None:
        try:
            corrected = agent.execute(source).corrected_sentence
        except Exception as exc:
            corrected = source
            error = str(exc)
            print(f"[{index}] LLM request failed: {exc}")

    if log_input_output:
        print(f"[{index}] INPUT : {source}")
        print(f"[{index}] OUTPUT: {corrected}")

    result = {
        "index": index,
        "doc_id": meta["doc_id"],
        "sent_id": meta["sent_id"],
        "source_tokenized": meta.get("source_tokenized"),
        "references_tokenized": meta.get("references_tokenized"),
        "source": source,
        "references": references,
        "corrected": corrected,
    }
    if error:
        result["error"] = error
    return result


def write_lines(lines: list[str], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def normalize_prediction(text: str) -> str:
    return " ".join(str(text).replace("\r", " ").replace("\n", " ").split())


def write_eval_ready_output(results: list[dict], path: Path) -> None:
    write_lines([normalize_prediction(r["corrected"]) for r in results], path)


def run_ua_evaluation(predictions_path: Path, m2_path: Path) -> dict:
    eval_script = repo_root() / "ua_evaluation" / "evaluation.py"
    if not m2_path.exists():
        print(f"Evaluation skipped: missing M2 file {m2_path}")
        return {"status": "skipped", "reason": f"missing M2 file {m2_path}"}

    cmd = [sys.executable, str(eval_script), str(predictions_path), "--m2", str(m2_path)]
    print("\n=== UA Evaluation ===")
    print(f"Running: {' '.join(cmd)}")
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if completed.stdout:
        print(completed.stdout.strip())
    if completed.stderr:
        print(completed.stderr.strip())

    return {
        "status": "ok" if completed.returncode == 0 else "failed",
        "reason": None,
        "command": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
    }


def count_m2_sentences(m2_path: Path) -> int:
    with open(m2_path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.startswith("S "))


def build_subset_m2(source_m2_path: Path, sent_ids: list[int], out_m2_path: Path) -> Path | None:
    if not source_m2_path.exists() or not sent_ids:
        return None

    with open(source_m2_path, "r", encoding="utf-8") as f:
        blocks = f.read().strip().split("\n\n")

    selected = []
    for sent_id in sent_ids:
        if sent_id < 1 or sent_id > len(blocks):
            return None
        selected.append(blocks[sent_id - 1].strip())

    with open(out_m2_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(selected).strip() + "\n")
    return out_m2_path


def resolve_m2_for_results(m2_path: Path, results: list[dict], run_dir: Path) -> Path:
    if not m2_path.exists() or count_m2_sentences(m2_path) == len(results):
        return m2_path

    print("Building aligned M2 for sliced run...")
    sent_ids = [int(r["sent_id"]) for r in results if r.get("sent_id") is not None]
    aligned_path = run_dir / "aligned_refs.m2"
    return build_subset_m2(m2_path, sent_ids, aligned_path) or m2_path


def slugify(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip()).strip("-")
    return slug or "unknown"


def run_name(config: dict) -> str:
    temperature = config.get("temperature")
    temp = f"temp{temperature:g}" if isinstance(temperature, (int, float)) else "tempNA"
    effort = f"effort{config['reasoning_effort']}" if config.get("reasoning_effort") else "effortNA"
    parts = [
        config.get("mode", "fast"),
        config.get("ua_partition", "valid"),
        config.get("model", "gpt-4.1-mini"),
        config.get("prompt_name", "base_en"),
        "single_pass",
        temp,
        effort,
    ]
    return "_".join(slugify(part) for part in parts)


def save_outputs(
    config: dict,
    config_path: Path,
    results: list[dict],
    m2_path: Path,
) -> None:
    run_dir = repo_root() / "outputs" / run_name(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / config_path.name)

    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": run_dir.name,
                "model": config.get("model"),
                "agent": config.get("agent", "gec_single_prompt"),
                "prompt": config.get("prompt_name", "base_en"),
                "partition": config.get("ua_partition"),
                "processing_mode": "single_pass",
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    write_lines([r["corrected"] for r in results], run_dir / "predictions.txt")
    eval_predictions_path = run_dir / "predictions.eval.txt"
    write_eval_ready_output(results, eval_predictions_path)

    eval_m2_path = resolve_m2_for_results(m2_path, results, run_dir)
    eval_result = run_ua_evaluation(eval_predictions_path, eval_m2_path)
    with open(run_dir / "evaluation.txt", "w", encoding="utf-8") as f:
        if eval_result.get("command"):
            f.write(f"Command: {' '.join(eval_result['command'])}\n")
        f.write(f"Status: {eval_result.get('status')}\n")
        if eval_result.get("reason"):
            f.write(f"Reason: {eval_result['reason']}\n")
        f.write("\n--- stdout ---\n")
        f.write(eval_result.get("stdout", "").strip() + "\n")
        f.write("\n--- stderr ---\n")
        f.write(eval_result.get("stderr", "").strip() + "\n")

    exact_matches = sum(
        1 for r in results if any(r["corrected"].strip() == ref.strip() for ref in r["references"])
    )
    exact_pct = (100 * exact_matches / len(results)) if results else 0.0
    with open(run_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"run_name: {run_dir.name}\n")
        f.write(f"exact_matches: {exact_matches}\n")
        f.write(f"total_samples: {len(results)}\n")
        f.write(f"exact_match_percent: {exact_pct:.1f}\n")

    print(f"Run directory: {run_dir}")
    print(f"Exact matches: {exact_matches}/{len(results)} ({exact_pct:.1f}%)")


def print_run_summary(config: dict, sample_count: int) -> None:
    fields = {
        "mode": config.get("mode", "fast"),
        "partition": config.get("ua_partition", "valid"),
        "model": config.get("model"),
        "agent": config.get("agent", "gec_single_prompt"),
        "prompt": config.get("prompt_name", "base_en"),
        "threads": config.get("num_threads", 10),
        "samples": sample_count,
    }
    print("=== Run ===")
    for key, value in fields.items():
        print(f"{key}: {value}")
    print("processing_mode: single_pass\n")


def main(config_path: str) -> None:
    config, resolved_config_path = load_config(config_path)
    sources, references, meta, m2_path = load_dataset(config)
    print_run_summary(config, len(sources))

    agent = build_agent(config)
    items = [
        (
            index,
            source,
            refs,
            item_meta,
            agent,
            bool(config.get("log_input_output", False)),
        )
        for index, (source, refs, item_meta) in enumerate(zip(sources, references, meta), start=1)
    ]
    results = process_in_parallel(
        items,
        process_sample,
        max_workers=int(config.get("num_threads", 10)),
        show_progress=bool(config.get("show_progress", True)),
    )

    save_outputs(config, resolved_config_path, results, m2_path)


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args.config)
    finally:
        cleanup_litellm_async_clients()
