"""Read immutable GitHub file versions, without executing repository code."""
import base64
import binascii
import hashlib
import json
import re
import time
from pathlib import PurePosixPath

import requests

from reposcouts.github.tools import GITHUB_API_BASE_URL, _get_headers

from .config import RagConfig
from .schemas import RepositoryFile, RepositorySnapshot
from .tracing import StageError, run_stage
from .security import sanitize_content


# 特殊字段匹配
_SHA = re.compile(r"^[0-9a-f]{40}$")
_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    "dist", "build", "vendor", "coverage", ".next", "target",
    "site-packages", ".idea", ".vscode",
}
_EXCLUDED_NAMES = {
    "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "poetry.lock",
    "uv.lock", "credentials.json", "secrets.json",
}
_DEPENDENCIES = {
    "pyproject.toml", "requirements.txt", "requirements-dev.txt",
    "package.json", "setup.py", "setup.cfg", "cargo.toml", "go.mod",
    "dockerfile", "compose.yaml", "docker-compose.yml",
}
_SUFFIXES = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".md", ".mdx", ".rst",
    ".toml", ".yaml", ".yml", ".json", ".ini", ".cfg", ".txt",
    ".go", ".rs", ".java", ".sh", ".sql",
}


class SourceError(StageError):
    """Messages are controlled codes, never raw HTTP bodies or credentials."""


def _get_json(session, path, config, deadline, *, params=None, max_bytes=1024 * 1024):
    return run_stage(
        "github.repository_read", {"path": path, "max_response_bytes": max_bytes},
        lambda: _request_json(session, path, config, deadline, params=params, max_bytes=max_bytes),
        lambda data: {"received": True}, tool=True,
    )


def _request_json(session, path, config, deadline, *, params=None, max_bytes=1024 * 1024):
    remaining = deadline - time.monotonic() # remaining time.
    if remaining <= 0:
        raise SourceError("repository_deadline")
    try:
        # 发送 session 请求
        with session.get(
            GITHUB_API_BASE_URL + path,
            headers=_get_headers(), params=params, stream=True,
            allow_redirects=False,
            timeout=(min(5.0, remaining), min(config.http_timeout_s, remaining)),
        ) as response:
            
            # 检查响应状态
            if response.status_code != 200:
                code = response.status_code
                raise SourceError(
                    "access_or_rate_limit" if code in (403, 429) else f"http_{code}"
                )

            # 创建容器
            body = bytearray() 
            # 按照块读取响应内容，直到达到最大字节数或超时
            for part in response.iter_content(chunk_size=16384): # 16 KiB per block; max_bytes caps the response.
                if time.monotonic() >= deadline:
                    raise SourceError("repository_deadline")

                body.extend(part) # add content to contains.
                if len(body) > max_bytes:
                    raise SourceError("response_too_large")

            return json.loads(body)
    except requests.RequestException:
        raise SourceError("network_error") from None
    except (ValueError, UnicodeError):
        raise SourceError("invalid_json") from None


def _eligible(entry):
    path = entry.get("path", "")
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
        return False

    # 对路径 path 中的每一部分进行校验，判断是否是相对路径
    parts = PurePosixPath(path).parts
    if not parts or PurePosixPath(path).is_absolute() or ".." in parts:
        return False

    # 路径小写，[-1] 取最后一部分即文件名
    lower_parts = [part.lower() for part in parts]
    name = lower_parts[-1]

    # 判断路径DIR、文件名、后缀
    if any(part in _EXCLUDED_DIRS for part in lower_parts[:-1]):
        return False
    if name.startswith(".env") or name in _EXCLUDED_NAMES:
        return False
    if name.endswith((".min.js", ".map", "_pb2.py", "_pb2_grpc.py", ".pem", ".key")):
        return False

    return (
        entry.get("type") == "blob"
        and entry.get("mode") in ("100644", "100755")
        and bool(_SHA.fullmatch(str(entry.get("sha", ""))))
        and type(entry.get("size")) is int
        and entry["size"] >= 0
        and (PurePosixPath(name).suffix in _SUFFIXES or name in _DEPENDENCIES)
    )


def _priority(entry):
    path = PurePosixPath(entry["path"])
    parts = [part.lower() for part in path.parts]
    name = parts[-1]
    if len(parts) == 1 and name.startswith("readme"):
        priority = 0
    elif name in _DEPENDENCIES:
        priority = 1
    elif any(part in ("test", "tests", "__tests__") for part in parts[:-1]):
        priority = 4
    elif path.suffix.lower() in (".md", ".mdx", ".rst"):
        priority = 3
    else:
        priority = 2
    return priority, len(parts), str(path).lower(), str(path)


def _read_blob(session, base, entry, config, deadline):
    # 从 Github 中读取 blob 内容
    data = _get_json(
        session, f"{base}/git/blobs/{entry['sha']}", config, deadline,
        max_bytes=2 * config.max_file_bytes + 8192,
    )

    try:
        if data.get("encoding") != "base64":
            raise SourceError("unsupported_encoding")

        # base64 解码 blob 内容
        encoded = "".join(data["content"].split())
        raw = base64.b64decode(encoded, validate=True)
    except (KeyError, TypeError, AttributeError, ValueError, binascii.Error):
        raise SourceError("invalid_blob") from None

    # 比较解码后的内容长度 与 文件数 entry 中记录的长度
    if len(raw) != entry["size"] or len(raw) > config.max_file_bytes:
        raise SourceError("blob_size_mismatch")

    # 计算解码后文件的 SHA，与 Github 中记录的 SHA 进行比较，确保文件内容未被篡改
    digest = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if digest != entry["sha"]:
        raise SourceError("blob_hash_mismatch")

    return raw


# 采集某个特点版本 repo 的文件和 metadata
def collect_repository_snapshot(full_name, config=None, *, session=None):
    config = config or RagConfig()
    # 检查 repo full_name 格式
    if not _REPO.fullmatch(full_name) or any(p in (".", "..") for p in full_name.split("/")):
        raise ValueError("full_name must be owner/repository")

    # 定义 snapshot 和 report 的实例
    snapshot = RepositorySnapshot(full_name=full_name)
    report = snapshot.report

    # session 是 HTTP 请求，owned_session 表示当前这个 session 是传入的还是当前内部创建的
    owned_session = session is None
    session = session if session is not None else requests.Session()

    base = f"/repos/{full_name}"
    deadline = time.monotonic() + config.repository_timeout_s

    # 定义跳过函数 记录跳过文件的个数以及原因
    def skip(reason):
        report.skipped_counts[reason] = report.skipped_counts.get(reason, 0) + 1

    try:
        # 先读取 repo 的最新一次 commit_sha，通过 commit_sha 读取对于 tree_sha 的文件列表
        commits = _get_json(session, base + "/commits", config, deadline, params={"per_page": 1})
        commit = commits[0]
        sha = commit["sha"]
        tree_sha = commit["commit"]["tree"]["sha"]
        if not _SHA.fullmatch(sha) or not _SHA.fullmatch(tree_sha):
            raise SourceError("invalid_commit")
        # 记录 RepoSnapshot 的 commit_sha
        snapshot.commit_sha = sha
        
        # 文件树读取，获取文件列表，限制了最大文件读取量
        tree = _get_json(
            session, f"{base}/git/trees/{tree_sha}", config, deadline,
            params={"recursive": "1"}, max_bytes=8 * 1024 * 1024,
        )
        entries = tree["tree"]
        if not isinstance(entries, list):
            raise SourceError("invalid_tree")

        report.tree_truncated = bool(tree.get("truncated"))
        report.discovered_files = sum(entry.get("type") == "blob" for entry in entries)
        eligible = sorted(
            (entry for entry in entries if _eligible(entry)),   # 判断文件是否需要采集
            key=_priority                                       # 按照定义的优先级进行采集
        )
        report.eligible_files = len(eligible)
        report.skipped_counts["filtered"] = report.discovered_files - len(eligible)

        # 计算总采集文件大小
        reserved_bytes = 0
        for entry in eligible:
            if time.monotonic() >= deadline:
                report.warnings.append("repository_deadline")
                break
            
            # 单文件大小
            if entry["size"] > config.max_file_bytes:
                skip("file_size_budget")
                continue
            
            # 总采样文件数量
            if report.selected_files >= config.max_files:
                skip("file_count_budget")
                continue
            
            # 敏感信息
            if sanitize_content(entry["path"]) != entry["path"]:
                skip("sensitive_path")
                continue
            
            # 总采集文件大小
            if reserved_bytes + entry["size"] > config.max_repository_bytes:
                skip("repository_size_budget")
                continue

            report.selected_files += 1
            reserved_bytes += entry["size"]
            try:
                # 读取文件并解码
                raw = _read_blob(session, base, entry, config, deadline)
                report.fetched_bytes += len(raw)
                if b"\0" in raw:
                    skip("binary")
                    continue
                
                try:
                    text = raw.decode("utf-8")
                except UnicodeError:
                    skip("non_utf8")
                    continue

                # 保存为 RepositoryFile 对象并存入 snapshot 中
                snapshot.files.append(RepositoryFile(
                    path=entry["path"], blob_sha=entry["sha"], content=text,
                ))
            except SourceError as exc:
                # record the failed file.
                report.failed_paths.append(entry["path"])
                report.warnings.append(str(exc))
                if str(exc) in ("access_or_rate_limit", "repository_deadline"):
                    break
    except SourceError as exc:
        report.warnings.append(str(exc))
    except (KeyError, IndexError, TypeError, AttributeError, ValueError):
        report.warnings.append("invalid_repository_response")
    finally:
        # 如果 session 请求是内部创建则关闭
        if owned_session:
            session.close()

    report.loaded_files = len(snapshot.files)
    limited = any(value for key, value in report.skipped_counts.items() if key != "filtered")
    report.status = (
        "failed" if not snapshot.files else
        "partial" if report.tree_truncated or report.failed_paths or report.warnings or limited else
        "ok"
    )
    return snapshot
