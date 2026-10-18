"""Fixed live baseline. Prepare first, then explicitly execute its ten cases once."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from decimal import Decimal
import hashlib
from importlib import metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import queue
import statistics
import subprocess
import sys
import threading
import time
import warnings
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reposcouts.observability.tracing import sanitize_trace_payload

CASES = ROOT / "tests/baselines/cases_v1.json"
OUTPUT = ROOT / "tests/artifacts/baselines"
ZONE = ZoneInfo("Asia/Shanghai")
SECRETS: tuple[str, ...] = ()


def now():
    return datetime.now(ZONE).isoformat()


def clean(value, limit=4000):
    return sanitize_trace_payload(value, secrets=SECRETS, max_string_chars=limit)


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else encode(value).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def emit(**values):
    print(encode(clean(values)), flush=True)


def initialize():
    global SECRETS
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    SECRETS = tuple(os.getenv(key, "") for key in ("DEEPSEEK_API_KEY", "GITHUB_TOKEN", "LANGSMITH_API_KEY"))

    class SafeFormatter(logging.Formatter):
        def format(self, record):
            return clean(super().format(record))

    handler = logging.StreamHandler()
    handler.setFormatter(SafeFormatter("%(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True)
    warnings.filterwarnings("ignore", message=r"(list_runs|get_run_url)\(\) is deprecated.*", category=DeprecationWarning)


def sources():
    paths = list((ROOT / "reposcouts").rglob("*.py")) + list((ROOT / "tests").glob("*.py"))
    paths += [ROOT / name for name in ("main.py", "requirements.txt", "requirements-dev.txt")]
    paths += [CASES, ROOT / "tests/baselines/README.md"]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path.read_bytes())
            for path in sorted(set(paths)) if path.is_file()}


def dependencies():
    return dict(sorted({item.metadata["Name"]: item.version for item in metadata.distributions()
                        if item.metadata.get("Name")}.items(), key=lambda pair: pair[0].lower()))


def git(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, encoding="utf-8")
    return result.stdout.strip() if result.returncode == 0 else None


def prepare():
    dataset = read(CASES)
    cases = dataset["cases"]
    if len(cases) != 10 or len({case["id"] for case in cases}) != 10 or len({case["request"] for case in cases}) != 10:
        raise ValueError("The fixed dataset must contain ten distinct cases and requests")
    if Counter(case["category"] for case in cases) != {"agent_rag": 6, "other": 4}:
        raise ValueError("Expected six Agent/RAG and four other cases")
    hashes = sources()
    packages = dependencies()
    source_hash = digest(hashes)
    baseline_id = datetime.now(ZONE).strftime("baseline-v1-%Y%m%d-%H%M%S")
    directory = OUTPUT / baseline_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "cases").mkdir()
    snapshot_files = []
    for relative, original_hash in hashes.items():
        data = (ROOT / relative).read_bytes().decode("utf-8")
        stored = clean(data, limit=4_000_000)
        target = directory / "code_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(stored, encoding="utf-8", newline="")
        snapshot_files.append({"path": relative, "original_sha256": original_hash,
                               "snapshot_sha256": digest(target.read_bytes()), "sanitized": stored != data})
    commit = git("rev-parse", "HEAD")
    manifest = {
        "baseline_id": baseline_id, "created_at": now(), "status": "prepared",
        "dataset_version": dataset["dataset_version"], "dataset_sha256": digest(dataset),
        "source_fingerprint": source_hash, "source_hashes": hashes, "snapshot_files": snapshot_files,
        "git_commit": commit, "git_branch": git("branch", "--show-current"),
        "git_status": git("status", "--porcelain", "--untracked-files=all", "--", "reposcouts", "tests/*.py", "tests/baselines", "main.py", "requirements*.txt"),
        "dependencies_sha256": digest(packages), "python_version": platform.python_version(),
        "python_executable": sys.executable, "platform": platform.platform(),
        "configuration": {
            "LANGSMITH_PROJECT": "reposcouts-baseline-v1",
            "REPOSCOUTS_TRACING_ENABLED": "true", "REPOSCOUTS_ENVIRONMENT": "baseline",
            "REPOSCOUTS_AGENT_VERSION": f"{commit or 'no-git'}-snapshot-{source_hash[:12]}",
            "REPOSCOUTS_PROMPT_VERSION": os.getenv("REPOSCOUTS_PROMPT_VERSION", "v1"),
            "DEEPSEEK_MODEL": os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        },
        "protocol": {"order": [case["id"] for case in cases], "repetitions": 1,
                     "automatic_case_reruns": False, "agent_internal_max_retries": 2,
                     "case_process_timeout_s": 900, "trace_readback_timeout_s": 90,
                     "content_policy": "No .env or raw LLM prompts/completions; redact configured secrets and token patterns; cap result strings at 4000 characters.",
                     "evidence_hash_scope": "Sanitized LangSmith tool inputs/outputs, not complete upstream GitHub files.",
                     "quality_policy": "Critic is an internal signal; independent human review remains pending."},
    }
    write(directory / "cases.json", dataset)
    write(directory / "dependencies.json", packages)
    (directory / "requirements-lock.txt").write_text("\n".join(f"{name}=={version}" for name, version in packages.items()) + "\n", encoding="utf-8")
    write(directory / "manifest.json", manifest)
    emit(stage="prepared", directory=str(directory), source_fingerprint=source_hash, cases=len(cases))


def known_sum(values, *, money=False):
    if any(value is None for value in values):
        return None
    return str(sum((Decimal(str(value)) for value in values), Decimal(0))) if money else sum(values)


def span_record(span):
    return {
        "id": str(span.id), "parent_run_id": str(span.parent_run_id) if span.parent_run_id else None,
        "name": span.name, "run_type": span.run_type, "start_time": span.start_time.isoformat(),
        "end_time": span.end_time.isoformat() if span.end_time else None,
        "latency_s": (span.end_time - span.start_time).total_seconds() if span.end_time else None,
        "input_tokens": span.prompt_tokens, "output_tokens": span.completion_tokens,
        "total_tokens": span.total_tokens,
        "cost_usd": str(span.total_cost) if span.total_cost is not None else None,
        "error": span.error,
    }


def trace_metrics(root, spans):
    llms = [span for span in spans if span.run_type == "llm"]
    tools = [span for span in spans if span.run_type == "tool"]
    return {
        "span_count": len(spans), "llm_calls": len(llms), "github_tool_calls": len(tools),
        "input_tokens": known_sum([span.prompt_tokens for span in llms]),
        "output_tokens": known_sum([span.completion_tokens for span in llms]),
        "total_tokens": known_sum([span.total_tokens for span in llms]),
        "cost_usd": known_sum([span.total_cost for span in llms], money=True),
        "llm_usage_coverage": {"calls": len(llms), "with_total_tokens": sum(span.total_tokens is not None for span in llms),
                               "with_cost": sum(span.total_cost is not None for span in llms)},
        "root_reported_tokens": root.total_tokens,
        "root_reported_cost_usd": str(root.total_cost) if root.total_cost is not None else None,
        "root_latency_s": (root.end_time - root.start_time).total_seconds() if root.end_time else None,
    }


def quality_signals(state):
    ranking = state.get("ranking") or {}
    top = ranking.get("top_repositories", [])
    names = [item["full_name"] for item in top]
    best = ranking.get("best_repository")
    evaluation = next((item for item in state.get("evaluations", []) if item["full_name"] == best), None)
    critic = state.get("critic_result") or {}
    return {
        "best_repository": best, "top_repositories": names,
        "best_in_top": best in names if ranking else None,
        "unique_top_repositories": len(names) == len(set(names)) if ranking else None,
        "candidate_count": len((state.get("candidates") or {}).get("candidates", [])),
        "evaluation_count": len(state.get("evaluations", [])),
        "best_must_have_checks": evaluation.get("must_have_checks", []) if evaluation else None,
        "critic_issue_counts": dict(Counter(item["severity"] for item in critic.get("issues", []))),
        "independent_quality_review": "pending",
    }


def collect_trace(reader, project, run_id, summary):
    from langsmith.utils import LangSmithNotFoundError
    deadline = time.monotonic() + 90
    previous = None
    roots, spans = [], []
    stable = False
    while time.monotonic() < deadline:
        try:
            roots = list(reader.list_runs(project_name=project, is_root=True, limit=3,
                filter='and(eq(metadata_key, "run_id"), eq(metadata_value, ' + json.dumps(run_id) + '))'))
        except LangSmithNotFoundError:
            roots = []
        if roots:
            spans = list(reader.list_runs(project_name=project, trace_id=roots[0].trace_id))
            signature = digest([span_record(span) for span in sorted(spans, key=lambda span: str(span.id))])
            ready = (roots[0].outputs or {}).get("observability") == summary and all(span.end_time for span in spans)
            if ready and signature == previous:
                stable = True
                break
            previous = signature if ready else None
        time.sleep(5)
    if not roots:
        return {"verified": False, "error": "No root trace found", "metrics": None}
    root = roots[0]
    payload = [{"inputs": span.inputs, "outputs": span.outputs, "extra": span.extra, "error": span.error} for span in spans]
    leaked = any(secret and secret in encode(payload) for secret in SECRETS)
    checks = {
        "one_root": len(roots) == 1, "stable_readback": stable,
        "all_spans_ended": bool(spans) and all(span.end_time is not None for span in spans),
        "summary_in_outputs": (root.outputs or {}).get("observability") == summary,
        "summary_in_metadata": summary is not None and all(root.extra.get("metadata", {}).get(key) == value for key, value in summary.items()),
        "no_configured_secrets": not leaked,
    }
    evidence = []
    for span in spans:
        if span.run_type == "tool":
            evidence.append({"name": span.name, "inputs": span.inputs,
                             "input_sha256": digest(span.inputs), "output_sha256": digest(span.outputs),
                             "repository_metadata": span.outputs if span.name == "get_repository_info" else None})
    node_names = {"parse_requirements", "search_candidates", "evaluate_repositories", "rank_repositories", "critic", "increment_retry"}
    return {
        "verified": all(checks.values()), "checks": checks,
        "trace_id": str(root.trace_id), "trace_url": reader.get_run_url(run=root, project_name=project),
        "metadata": root.extra.get("metadata", {}), "metrics": trace_metrics(root, spans),
        "node_invocations": [span_record(span) for span in sorted(spans, key=lambda span: span.start_time)
                             if span.parent_run_id == root.id and span.name in node_names],
        "spans": [span_record(span) for span in spans], "github_evidence": evidence,
    }


def worker(directory, case_id):
    from fastapi.testclient import TestClient
    from langsmith import Client
    manifest = read(directory / "manifest.json")
    os.environ.update(manifest["configuration"])
    # Model clients are created on import, after the frozen model configuration is set.
    import reposcouts.api.app as api
    from reposcouts.api.run_manager import TERMINAL_EVENT_TYPES
    case = next(case for case in read(directory / "cases.json")["cases"] if case["id"] == case_id)
    path = directory / "cases" / f"{case_id}.json"
    if path.exists():
        raise RuntimeError("Case already attempted; automatic reruns are forbidden")
    result = {"case": case, "attempts": 1, "started_at": now(), "collection_status": "running",
              "events": [], "retry_trajectory": [], "state": {}, "trace": None}
    write(path, result)
    started = time.perf_counter()
    stage = "workflow"
    reader = None
    try:
        with TestClient(api.app) as http:
            response = http.post("/api/runs", json={"user_request": case["request"]})
            response.raise_for_status()
            run_id = response.json()["run_id"]
            record = api.run_manager.get_run(run_id)
            result["business_run_id"] = run_id
            write(path, result)
            emit(stage="case-started", case=case_id, business_run_id=run_id)
            while True:
                try:
                    event = record.events.get(timeout=20)
                except queue.Empty:
                    emit(stage="case-heartbeat", case=case_id, elapsed_s=round(time.perf_counter() - started, 1), status=record.status)
                    continue
                elapsed = round(time.perf_counter() - started, 3)
                entry = {"type": event.type, "node": event.node, "elapsed_s": elapsed}
                if event.type == "node.completed":
                    output = event.data.get("output", {})
                    result["state"].update(output)
                    if event.node == "critic":
                        result["retry_trajectory"].append({"kind": "critic", "elapsed_s": elapsed, "result": output.get("critic_result")})
                if event.type == "retry.started":
                    result["retry_trajectory"].append({"kind": "retry", "elapsed_s": elapsed, **event.data})
                result["events"].append(entry)
                if event.type == "node.completed" or event.type in TERMINAL_EVENT_TYPES:
                    write(path, result)
                    emit(stage="case-event", case=case_id, **entry)
                if event.type in TERMINAL_EVENT_TYPES:
                    result["terminal_event"] = event.type
                    result["terminal_payload"] = event.data
                    result["workflow_wall_time_s"] = elapsed
                    break
            for thread in threading.enumerate():
                if thread.name == f"reposcout-run-{run_id[:8]}":
                    thread.join(timeout=15)
            result["summary"] = record.summary
            result["quality_signals"] = quality_signals(result["state"])
            write(path, result)
        stage = "trace-readback"
        reader = Client(timeout_ms=(5000, 15000))
        result["trace"] = collect_trace(reader, manifest["configuration"]["LANGSMITH_PROJECT"], run_id, record.summary)
        result["collection_status"] = "finished"
    except Exception as exc:
        result["collection_status"] = "collection_error"
        result["collection_error"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
    finally:
        if reader is not None:
            reader.close(timeout=0)
        result["finished_at"] = now()
        result["collection_wall_time_s"] = round(time.perf_counter() - started, 3)
        write(path, result)
    metrics = (result.get("trace") or {}).get("metrics") or {}
    emit(stage="case-finished", case=case_id, summary=result.get("summary"),
         trace_verified=(result.get("trace") or {}).get("verified", False), **metrics)
    return 0 if result["collection_status"] == "finished" else 1


def aggregate(directory):
    manifest = read(directory / "manifest.json")
    rows = []
    for case in read(directory / "cases.json")["cases"]:
        path = directory / "cases" / f"{case['id']}.json"
        result = read(path) if path.exists() else {}
        trace = result.get("trace") or {}
        metric = trace.get("metrics") or {}
        summary = result.get("summary") or {}
        rows.append({"id": case["id"], "title": case["title"], "attempted": bool(result),
                     "execution_status": summary.get("execution_status"), "outcome": summary.get("outcome"),
                     "critic_passed": summary.get("critic_passed"), "retry_count": summary.get("retry_count"),
                     "wall_time_s": result.get("workflow_wall_time_s"), "llm_calls": metric.get("llm_calls"),
                     "tool_calls": metric.get("github_tool_calls"), "total_tokens": metric.get("total_tokens"),
                     "cost_usd": metric.get("cost_usd"), "trace_verified": trace.get("verified", False),
                     "best_repository": (result.get("quality_signals") or {}).get("best_repository"),
                     "trace_url": trace.get("trace_url"), "collection_status": result.get("collection_status")})
    latencies = [row["wall_time_s"] for row in rows if row["wall_time_s"] is not None]
    summary = {
        "baseline_id": manifest["baseline_id"], "status": manifest["status"],
        "attempted_cases": sum(row["attempted"] for row in rows), "expected_cases": 10,
        "trace_verified_cases": sum(row["trace_verified"] for row in rows),
        "execution_status_counts": dict(Counter(row["execution_status"] or "unknown" for row in rows)),
        "outcome_counts": dict(Counter(row["outcome"] or "unknown" for row in rows)),
        "total_tokens": known_sum([row["total_tokens"] for row in rows]),
        "total_cost_usd": known_sum([row["cost_usd"] for row in rows], money=True),
        "known_cost_cases": sum(row["cost_usd"] is not None for row in rows),
        "latency_mean_s": statistics.mean(latencies) if latencies else None,
        "latency_median_s": statistics.median(latencies) if latencies else None,
        "latency_p95_s": sorted(latencies)[math.ceil(len(latencies) * .95) - 1] if latencies else None,
        "rows": rows,
    }
    write(directory / "summary.json", summary)
    lines = [f"# {manifest['baseline_id']}", "", f"状态：{manifest['status']}。每条需求仅执行一次；失败结果保留，不补跑。", "",
             "|需求|执行状态 / outcome|秒|LLM|Tool|Token|USD|最佳仓库|", "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        display = lambda value: "—" if value is None else str(value)
        lines.append("|" + "|".join(display(value) for value in (
            row["title"], f"{row['execution_status']} / {row['outcome']}", row["wall_time_s"], row["llm_calls"],
            row["tool_calls"], row["total_tokens"], row["cost_usd"], row["best_repository"])) + "|")
    lines += ["", "费用是 LangSmith 估算，未知为 null（表格显示 —），不按零费用处理。", "",
              "Critic 通过是内部信号，独立质量评审尚未完成；十个不同输入各一次不能衡量同一输入的随机波动。", "",
              "代码与依赖见 manifest.json、code_snapshot/、dependencies.json 和 requirements-lock.txt。",
              "GitHub 数据会变化；保留查询、候选、元数据和已脱敏工具输出的 SHA256，不能保证外部数据完全重放。", "",
              "首次快照可能包含未提交代码；以 source_fingerprint 和每个文件的 original_sha256 标识实际版本。",
              "源码副本若触发脱敏，snapshot_files 会标记 sanitized=true；未保存 .env 或真实密钥。", "",
              "## Trace 链接", ""]
    lines += [f"- {row['id']}: {row['trace_url'] or '未获取'}" for row in rows]
    (directory / "BASELINE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def execute(directory):
    from langsmith import Client
    manifest = read(directory / "manifest.json")
    if manifest["status"] != "prepared":
        raise RuntimeError("This batch has already started; create a new batch instead of repeating cases")
    if not all(SECRETS):
        raise RuntimeError("Missing required credentials")
    reader = Client(timeout_ms=(5000, 15000))
    try:
        list(reader.list_projects(limit=1))
    finally:
        reader.close(timeout=0)
    manifest["status"] = "running"
    manifest["started_at"] = now()
    write(directory / "manifest.json", manifest)
    try:
        for index, case in enumerate(read(directory / "cases.json")["cases"], 1):
            if sources() != manifest["source_hashes"] or digest(dependencies()) != manifest["dependencies_sha256"]:
                raise RuntimeError("Source or dependency drift detected; batch stopped")
            emit(stage="batch-progress", case=index, total=10, case_id=case["id"])
            path = directory / "cases" / f"{case['id']}.json"
            try:
                completed = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                    "--worker", str(directory), "--case-id", case["id"]], cwd=ROOT, timeout=900)
                if completed.returncode and not path.exists():
                    write(path, {"case": case, "attempts": 1, "collection_status": "worker_error", "returncode": completed.returncode})
            except subprocess.TimeoutExpired:
                partial = read(path) if path.exists() else {"case": case, "attempts": 1}
                partial["collection_status"] = "process_timeout"
                write(path, partial)
            aggregate(directory)
        if sources() != manifest["source_hashes"] or digest(dependencies()) != manifest["dependencies_sha256"]:
            raise RuntimeError("Source or dependency drift detected at end of batch")
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "interrupted"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        manifest["finished_at"] = now()
        write(directory / "manifest.json", manifest)
        summary = aggregate(directory)
        emit(stage="batch-finished", directory=str(directory), status=manifest["status"],
             attempted=summary["attempted_cases"], verified=summary["trace_verified_cases"],
             total_tokens=summary["total_tokens"], total_cost_usd=summary["total_cost_usd"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--execute", type=Path)
    mode.add_argument("--worker", type=Path)
    parser.add_argument("--case-id")
    args = parser.parse_args()
    initialize()
    if args.prepare:
        prepare()
        return 0
    directory = (args.execute or args.worker).resolve()
    if not directory.is_relative_to(OUTPUT.resolve()):
        raise ValueError("Baseline output must be inside tests/artifacts/baselines")
    if args.execute:
        execute(directory)
        return 0
    return worker(directory, args.case_id)


if __name__ == "__main__":
    raise SystemExit(main())
