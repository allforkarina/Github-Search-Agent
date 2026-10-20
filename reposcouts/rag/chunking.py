"""Partition original source lines; token counting is injected by the embedder."""
import ast
import hashlib
import re
from collections.abc import Callable
from pathlib import PurePosixPath

from .config import RagConfig
from .schemas import ChunkingResult, EvidenceChunk, RepositorySnapshot
from .semantic import semantic_sections


def chunk_identity(full_name, commit_sha, path, start, end, text):
    identity = "\0".join((full_name, commit_sha, path, str(start), str(end), text))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def format_passage(path: str, symbol: str | None, text: str) -> str:
    """
    Example passage: 
    reposcouts/search.py
    search_repo
    def search_repo(query):
        return search(query)
    """
    return f"passage: {path}\n{symbol or ''}\n{text}"


def _python_sections(text, line_count):
    """
    Chunk Python source code into sections based on top-level classes and functions.
    """
    # Python 源码解析为抽象语法树(AST) 识别类、函数、代码范围
    tree = ast.parse(text)
    spans = []

    def visit(node, prefix=""):
        for child in getattr(node, "body", []):
            # 判断是否是 AST 类、函数、异步函数定义
            if isinstance(
                child, 
                (
                    ast.ClassDef, 
                    ast.FunctionDef, 
                    ast.AsyncFunctionDef
                )
            ):
                symbol = f"{prefix}.{child.name}" if prefix else child.name
                # 每一个结构的起始行号
                start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                # 生成对应的代码块区域
                spans.append((start, child.end_lineno, symbol))
                # 递归子节点 有限最小结构如异步函数 > 函数 > 类
                visit(child, symbol)

    # 从 AST 树的根节点开始遍历 获取代码块span
    visit(tree)
    boundaries = {1, line_count + 1}
    for start, end, _ in spans:
        # 将代码块的起始行和结束行加入边界集合
        boundaries.update((start, end + 1))

    # 每一个结构的起始、结尾对应的行号，形成一个列表并排序
    points = sorted(boundaries)
    sections = []
    # 每一次都取相邻两个行号，不一定是完整的结构，需要判断 active
    for start, next_start in zip(points, points[1:]):
        # 判断当前 start 处于哪些结构的代码块，可以是类，也可以是类中的函数，甚至是 None
        active = [span for span in spans if span[0] <= start <= span[1]]
        # 取最小的结构，即最内层的结构作为当前 chunk 的 symbol
        symbol = min(active, key=lambda span: span[1] - span[0])[2] if active else None
        sections.append((start, next_start - 1, symbol))
    return sections


def _markdown_sections(lines):
    starts = [(1, None)]    # 每一个 section 的起始行号 以及对应的标题
    headings = []           # 当前的标题栈，存储当前层级的标题信息
    fence = None            # 表示当前不在代码框中

    # 遍历每一行，寻找标题和代码框的标记
    for number, line in enumerate(lines, 1):
        # 在代码框
        if fence is not None:
            # 检查当前行是不是代码框的结束标记，如果是则退出代码框状态
            if re.fullmatch(r" {0,3}" + re.escape(fence[0]) + "{" + str(fence[1]) + r",}[ \t]*\r?\n?", line):
                fence = None
            continue
        
        # 判断是否是代码框 ``` 或者 ~~~
        match = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if match:
            marker = match.group(1)
            # 代码框的结束标记
            fence = (marker[0], len(marker))
            continue
        
        # 判断是否是标题行，标题行的格式为 # 或 ## 或 ### 等，后面跟着标题文本
        match = re.match(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*$", line)
        heading_start = number
        if match:
            level = len(match.group(1))
            title = re.sub(r"[ \t]+#+$", "", match.group(2))
        elif (number > 1 and re.fullmatch(r" {0,3}(?:=+|-+)[ \t]*\r?\n?", line)
              and lines[number - 2].strip()
              and not re.match(r"\s*(?:[#>`~]|[-*+]\s)", lines[number - 2])):
            heading_start = number - 1
            level = 1 if line.lstrip().startswith("=") else 2
            title = lines[number - 2].strip()
        else:
            continue
        if title:
            headings = [(depth, title_) for depth, title_ in headings if depth < level]
            headings.append((level, title))
            symbol = " > ".join(title_ for _, title_ in headings)
            if heading_start == 1:
                starts[0] = (1, symbol)
            else:
                starts.append((heading_start, symbol))
    return [
        (start, starts[index + 1][0] - 1 if index + 1 < len(starts) else len(lines), symbol)
        for index, (start, symbol) in enumerate(starts)
    ]


def chunk_snapshot(
    snapshot: RepositorySnapshot,
    *,
    token_count: Callable[[str], int],
    config: RagConfig | None = None,
) -> ChunkingResult:
    
    config = config or RagConfig()
    result = ChunkingResult()
    
    # 没有创建 Repo Snapshot，返回空实例
    if not snapshot.files:
        return result
    if snapshot.commit_sha is None:
        raise ValueError("a snapshot with files must have a commit SHA")

    # 对 repo 中的合法文件逐文件处理
    for file in snapshot.files:
        # 行分割 保留行号和换行符
        lines = file.content.splitlines(keepends=True)
        if not lines:
            continue
        
        # 分析文件后缀，选择不同的分块策略
        suffix = PurePosixPath(file.path).suffix.lower()
        if suffix == ".py":
            try:
                sections = _python_sections(file.content, len(lines))
            except (SyntaxError, ValueError, RecursionError):
                result.fallback_files.append(file.path)
                sections = [(1, len(lines), None)]

        elif suffix in (".md", ".mdx"):
            sections = _markdown_sections(lines)

        elif suffix in (".js", ".jsx", ".ts", ".tsx", ".java"):
            try:
                sections = semantic_sections(file.content, suffix, len(lines))
            except (ImportError, ValueError, TypeError, RecursionError):
                result.fallback_files.append(file.path)
                sections = [(1, len(lines), None)]

        else:
            # 普通文本按照行数分块
            sections = [(1, len(lines), None)]

        # 对分块后的区域 按照 token_count and overlap_tokens 切分
        for first, last, symbol in sections:
            # 当前 chunk 的起始行号
            start = first - 1
            while start < last:
                end = start
                # 移动 chunk 的end
                while end < last:
                    # 逐行添加 passage，如果 token_count 超过 chunk_tokens 则停止
                    passage = format_passage(file.path, symbol, "".join(lines[start:end + 1]))
                    if token_count(passage) > config.chunk_tokens:
                        break
                    end += 1

                if end == start:
                    # 当前一行的 token_count 已经超过 chunk_tokens，记录为 oversized_lines
                    result.oversized_lines.setdefault(file.path, []).append(start + 1)
                    # 重新选择 chunk 的起始行号
                    start += 1
                    continue
                
                # 提取当前 chunk 的文本
                text = "".join(lines[start:end])
                if text.strip():
                    # text identity with name and sha.
                    result.chunks.append(EvidenceChunk(
                        chunk_id=chunk_identity(snapshot.full_name, snapshot.commit_sha, file.path,
                                                start + 1, end, text),
                        full_name=snapshot.full_name, commit_sha=snapshot.commit_sha,
                        path=file.path, symbol=symbol,
                        start_line=start + 1, end_line=end, text=text,
                    ))
                    
                # 当前区域切分完毕
                if end == last:
                    break
                
                # 考虑 overlap_tokens 的情况，寻找下一个 chunk 的起始行号
                next_start = end
                for overlap_start in range(end - 1, start, -1):
                    if token_count("".join(lines[overlap_start:end])) > config.overlap_tokens:
                        break
                    next_start = overlap_start
                start = next_start
    return result
