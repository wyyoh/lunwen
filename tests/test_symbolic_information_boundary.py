from __future__ import annotations

import ast
from pathlib import Path

from keyed_gram.authsynth_symbolic_shared import VerificationKind, VerificationResult
from keyed_gram.stage_f2c_protocol import information_boundary_audit


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_analyzer_package_has_no_verifier_or_evaluator_dependency() -> None:
    analyzer = _root() / "src/keyed_gram/authsynth_symbolic_analyzer"
    forbidden = ("authsynth_symbolic_verifier", "authsynth_symbolic_evaluator")
    for path in analyzer.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        modules.extend(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any(part in module for part in forbidden for module in modules)


def test_analyzer_has_no_hidden_or_gold_symbols_and_no_fixed_catalog() -> None:
    audit = information_boundary_audit(_root(), ())
    assert audit["status"] == "passed"
    assert audit["analyzer_forbidden_symbol_failure_count"] == 0
    assert audit["analyzer_fixed_query_catalog_access"] is False


def test_verifier_result_schema_cannot_return_hidden_formula() -> None:
    result = VerificationResult(VerificationKind.EQUIVALENT, reason_code="equivalent")
    assert set(result.to_dict()) == {
        "kind",
        "assignment",
        "counterexample_digest",
        "observed_version_digest",
        "reason_code",
    }
    assert "formula" not in result.to_dict()


def test_public_analyzer_schema_does_not_contain_evaluator_labels() -> None:
    from f2c_symbolic_helpers import case_for

    public = case_for("hidden_guarded_effect").analyzer_input.to_public_dict()
    forbidden = {
        "mutation_category",
        "omission_atoms",
        "reference_contract",
        "gold_contract",
        "hidden_implementation",
        "expected_unknown",
    }
    assert not (set(public) & forbidden)
