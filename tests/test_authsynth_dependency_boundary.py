from __future__ import annotations

import ast
from pathlib import Path


def test_analyzer_has_no_evaluator_or_hidden_benchmark_dependency() -> None:
    root = Path(__file__).resolve().parents[1]
    analyzer = root / "src/keyed_gram/authsynth_analyzer"
    forbidden_fragments = (
        "authsynth_evaluator",
        "implementation_map",
        "mutation_category",
        "omission_type",
        "expected_spurious_counterexample",
        "gold_shield",
        "gold_contract",
    )
    for path in analyzer.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        imports.extend(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any("authsynth_evaluator" in item for item in imports)
        lowered = source.casefold()
        assert not any(item in lowered for item in forbidden_fragments)
