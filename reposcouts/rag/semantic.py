"""Read-only Tree-sitter adapters. Never import or execute candidate code."""
from importlib import import_module
from importlib.metadata import version, PackageNotFoundError


def parser_fingerprint():
    parts = ["semantic-v2"]
    for name in ("tree-sitter", "tree-sitter-java", "tree-sitter-javascript", "tree-sitter-typescript"):
        try:
            parts.append(name + "@" + version(name))
        except PackageNotFoundError:
            parts.append(name + "@missing")
    return ";".join(parts)


def partition_spans(spans, line_count):
    points = sorted({1, line_count + 1, *(start for start, _, _ in spans),
                     *(end + 1 for _, end, _ in spans)})
    result = []
    for start, next_start in zip(points, points[1:]):
        active = [span for span in spans if span[0] <= start <= span[1]]
        symbol = min(active, key=lambda span: (span[1] - span[0], -span[2].count(".")))[2] if active else None
        result.append((start, next_start - 1, symbol))
    return result


def semantic_sections(text, suffix, line_count):
    from tree_sitter import Language, Parser

    module, factory = {
        ".js": ("tree_sitter_javascript", "language"),
        ".jsx": ("tree_sitter_javascript", "language"),
        ".ts": ("tree_sitter_typescript", "language_typescript"),
        ".tsx": ("tree_sitter_typescript", "language_tsx"),
        ".java": ("tree_sitter_java", "language"),
    }[suffix]
    parser = Parser(Language(getattr(import_module(module), factory)()))
    raw = text.encode("utf-8")
    tree = parser.parse(raw)
    if tree is None or tree.root_node.has_error:
        raise ValueError("semantic_parse_failed")
    declarations = {
        "class_declaration", "abstract_class_declaration", "function_declaration",
        "generator_function_declaration", "method_definition", "method_declaration",
        "constructor_declaration", "interface_declaration", "enum_declaration",
        "record_declaration", "annotation_type_declaration", "type_alias_declaration",
    }
    spans = []

    def visit(node, prefix=""):
        name_node = node.child_by_field_name("name")
        is_arrow = node.type == "variable_declarator" and any(
            child.type in ("arrow_function", "function_expression", "generator_function")
            for child in node.named_children
        )
        symbol = prefix
        if (node.type in declarations or is_arrow) and name_node is not None:
            name = raw[name_node.start_byte:name_node.end_byte].decode("utf-8")
            symbol = f"{prefix}.{name}" if prefix else name
            first = node.start_point.row + 1
            last = node.end_point.row + (1 if node.end_point.column else 0)
            spans.append((first, max(first, last), symbol))
        for child in node.named_children:
            visit(child, symbol)

    visit(tree.root_node)
    return partition_spans(spans, line_count)
