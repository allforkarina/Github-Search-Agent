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


_SHA = re.compile(r"^[0-9a-f]{40}$")
_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    "dist", "build", "vendor", "coverage", ".next", "target",
    "site-packages", ".idea", ".vscode",
    ".gradle", ".m2",
}
_EXCLUDED_NAMES = {
    "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "poetry.lock",
    "uv.lock", "credentials.json", "secrets.json",
    "settings-security.xml", "settings.xml",
}
_DEPENDENCIES = {
    "pyproject.toml", "requirements.txt", "requirements-dev.txt",
    "package.json", "setup.py", "setup.cfg", "cargo.toml", "go.mod",
    "dockerfile", "compose.yaml", "docker-compose.yml",
    "pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle", "gradle.properties",
}
_SUFFIXES = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".md", ".mdx", ".rst",
    ".toml", ".yaml", ".yml", ".json", ".ini", ".cfg", ".txt",
    ".go", ".rs", ".java", ".sh", ".sql",
    ".xml", ".gradle", ".properties",
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
        # github repo info fetch.
        with session.get(
            GITHUB_API_BASE_URL + path,
            headers=_get_headers(), params=params, stream=True,
            allow_redirects=False,
            timeout=(min(5.0, remaining), min(config.http_timeout_s, remaining)),
        ) as response:
            if response.status_code != 200:
                code = response.status_code
                raise SourceError(
                    "access_or_rate_limit" if code in (403, 429) else f"http_{code}"
                )

            body = bytearray() # contains.
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
    # validation str and relative path.
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
        return False

    parts = PurePosixPath(path).parts
    if not parts or PurePosixPath(path).is_absolute() or ".." in parts:
        return False

    # lowercase
    lower_parts = [part.lower() for part in parts]
    name = lower_parts[-1]

    # exclude dirs and files.
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
    elif name in _DEPENDENCIES and len(parts) == 1:
        priority = 1
    elif any(part in ("test", "tests", "__tests__") for part in parts[:-1]):
        priority = 4
    elif path.suffix.lower() in (".md", ".mdx", ".rst"):
        priority = 3
    else:
        priority = 2
    return priority, len(parts), str(path).lower(), str(path)


def _read_blob(session, base, entry, config, deadline):
    data = _get_json(
        session, f"{base}/git/blobs/{entry['sha']}", config, deadline,
        max_bytes=2 * config.max_file_bytes + 8192,
    )

    try:
        if data.get("encoding") != "base64":
            raise SourceError("unsupported_encoding")

        # extract and decode the base64 content of the blob.
        encoded = "".join(data["content"].split())
        raw = base64.b64decode(encoded, validate=True)
    except (KeyError, TypeError, AttributeError, ValueError, binascii.Error):
        raise SourceError("invalid_blob") from None

    if len(raw) != entry["size"] or len(raw) > config.max_file_bytes:
        raise SourceError("blob_size_mismatch")

    # verify blob hash.
    digest = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if digest != entry["sha"]:
        raise SourceError("blob_hash_mismatch")

    return raw


def collect_repository_snapshot(full_name, config=None, *, session=None, queries=None, commit_sha=None):
    config = config or RagConfig()
    if not _REPO.fullmatch(full_name) or any(p in (".", "..") for p in full_name.split("/")):
        raise ValueError("full_name must be owner/repository")

    # collect the repo file and get the report.
    snapshot = RepositorySnapshot(full_name=full_name)
    report = snapshot.report

    owned_session = session is None
    session = session if session is not None else requests.Session()

    base = f"/repos/{full_name}"
    deadline = time.monotonic() + config.repository_timeout_s
    snapshot._deadline = deadline

    def skip(reason):
        report.skipped_counts[reason] = report.skipped_counts.get(reason, 0) + 1

    try:
        # The list endpoint supplies the head commit and its tree without file diffs.
        if commit_sha is not None:
            if not _SHA.fullmatch(commit_sha):
                raise SourceError("invalid_commit")
            raw_commit = _get_json(session, base + "/git/commits/" + commit_sha, config, deadline)
            commit = {"sha": raw_commit["sha"], "commit": {"tree": raw_commit["tree"]}}
            if commit["sha"] != commit_sha:
                raise SourceError("commit_mismatch")
        else:
            commits = _get_json(session, base + "/commits", config, deadline, params={"per_page": 1})
            commit = commits[0]
        sha = commit["sha"]
        tree_sha = commit["commit"]["tree"]["sha"]
        if not _SHA.fullmatch(sha) or not _SHA.fullmatch(tree_sha):
            raise SourceError("invalid_commit")

        snapshot.commit_sha = sha
        tree = _get_json(
            session, f"{base}/git/trees/{tree_sha}", config, deadline,
            params={"recursive": "1"}, max_bytes=8 * 1024 * 1024,
        )
        entries = tree["tree"]
        if not isinstance(entries, list):
            raise SourceError("invalid_tree")

        report.tree_truncated = bool(tree.get("truncated"))
        # dicovered total files count.
        report.discovered_files = sum(entry.get("type") == "blob" for entry in entries)
        # check whether fit the criteria and count.
        eligible = sorted((entry for entry in entries if _eligible(entry)), key=_priority)
        snapshot.available_files = eligible
        if queries:
            from .retrieval import lexical_tokens
            terms = set(lexical_tokens(" ".join(queries)))
            eligible = sorted(eligible, key=lambda e: (
                _priority(e)[0] > 1, -len(set(lexical_tokens(e["path"])) & terms), _priority(e),
            ))
        # total eligible files count.
        report.eligible_files = len(eligible)
        # total skipped files count.(Total - fit_criteria)
        report.skipped_counts["filtered"] = report.discovered_files - len(eligible)

        # count total bytes.
        reserved_bytes = 0
        for entry in eligible:
            if time.monotonic() >= deadline:
                report.warnings.append("repository_deadline")
                break
            if entry["size"] > config.max_file_bytes:
                skip("file_size_budget")
                continue
            if report.selected_files >= min(config.max_files, config.initial_files):
                skip("file_count_budget")
                continue
            if sanitize_content(entry["path"]) != entry["path"]:
                skip("sensitive_path")
                continue
            if reserved_bytes + entry["size"] > min(config.max_repository_bytes, config.initial_repository_bytes):
                skip("repository_size_budget")
                continue

            # Reserve before fetching: failed attempts also consume the request budget.
            report.selected_files += 1
            reserved_bytes += entry["size"]
            report.reserved_bytes = reserved_bytes
            report.attempted_paths.append(entry["path"])
            try:
                raw = _read_blob(session, base, entry, config, deadline)
                report.fetched_bytes += len(raw)
                # Check for null bytes.
                if b"\0" in raw:
                    skip("binary")
                    continue
                try:
                    text = raw.decode("utf-8")
                except UnicodeError:
                    skip("non_utf8")
                    continue

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


def expand_repository_snapshot(snapshot, paths, config, *, session=None):
    """Fetch only pinned tree blobs, sharing initial-round attempts/bytes/deadline."""
    if not snapshot.commit_sha or not snapshot.available_files:
        return 0
    report = snapshot.report
    deadline = (snapshot._deadline if snapshot._deadline is not None
                else time.monotonic() + config.repository_timeout_s)
    lookup = {entry["path"]: entry for entry in snapshot.available_files}
    attempted = set(report.attempted_paths) | {file.path for file in snapshot.files}
    owned = session is None
    session = session if session is not None else requests.Session()
    loaded, attempts = 0, 0
    try:
        for path in dict.fromkeys(paths):
            if time.monotonic() >= deadline:
                report.warnings.append("repository_deadline")
                break
            if report.selected_files >= config.max_files or attempts >= config.expansion_batch_size:
                break
            entry = lookup.get(path)
            if (entry is None or path in attempted or not _eligible(entry)
                    or sanitize_content(path) != path or entry["size"] > config.max_file_bytes
                    or report.reserved_bytes + entry["size"] > config.max_repository_bytes):
                continue
            attempted.add(path)
            report.attempted_paths.append(path)
            report.selected_files += 1
            report.reserved_bytes += entry["size"]
            attempts += 1
            try:
                raw = _read_blob(session, f"/repos/{snapshot.full_name}", entry, config, deadline)
                report.fetched_bytes += len(raw)
                if b"\0" in raw:
                    continue
                text = raw.decode("utf-8")
                snapshot.files.append(RepositoryFile(path=path, blob_sha=entry["sha"], content=text))
                loaded += 1
            except UnicodeError:
                report.warnings.append("non_utf8")
            except SourceError as exc:
                report.failed_paths.append(path)
                report.warnings.append(str(exc))
                if str(exc) in ("access_or_rate_limit", "repository_deadline"):
                    break
    finally:
        if owned:
            session.close()
    report.expansion_rounds += 1
    report.loaded_files = len(snapshot.files)
    # Initial omissions remain reported even if some are recovered in later rounds.
    if report.loaded_files < report.eligible_files or report.tree_truncated or report.warnings or report.failed_paths:
        report.status = "partial" if snapshot.files else "failed"
    else:
        report.status = "ok"
    return loaded
