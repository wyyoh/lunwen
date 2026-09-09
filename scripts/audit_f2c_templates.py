"""仅审计生成器模板定义，绝不 materialize 或评分正式 held-out case。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from keyed_gram.authsynth_symbolic_shared import canonical_digest
from keyed_gram.authsynth_symbolic_verifier.benchmark import (
    CONTROLS,
    MUTATIONS,
    SPLITS,
    _legacy_collision_audit,
)
from keyed_gram.authsynth_symbolic_verifier.template_families import definition
from keyed_gram.authsynth_symbolic_verifier.template_isolation import audit_templates


def run_audit():
    templates = tuple(definition(s, c) for s in SPLITS for c in (*MUTATIONS, *CONTROLS))
    audit = audit_templates(templates)
    # 旧审计仍然执行；适配的是模板 metadata，不是 HiddenSymbolicCase。
    metadata = []
    for template in templates:
        reference = {
            "clauses": [c.to_dict() for c in template.contract_clauses],
            "state_updates": [u.to_dict() for u in template.contract_updates],
        }
        metadata.append(
            SimpleNamespace(
                split=template.split,
                tool_family=template.family,
                mutation_family=template.family,
                guard_template_family=template.family,
                field_binding_family=template.family,
                version_lineage=template.family,
                ast_shape_family=template.family,
                public_case_id=template.family,
                analyzer_input=SimpleNamespace(schema=template.schema),
                implementation=SimpleNamespace(
                    transitions=template.transitions,
                    loop=template.loop,
                    after_loop=template.after_loop,
                ),
                reference_contract=SimpleNamespace(to_dict=lambda r=reference: r),
            )
        )
    legacy = _legacy_collision_audit(metadata)
    audit["legacy_content_audit"] = legacy
    audit["template_family_counts_by_split"] = dict(Counter(t.split for t in templates))
    audit["template_definition_digests"] = [
        canonical_digest(
            {
                "schema": t.schema.to_dict(),
                "transitions": [r.to_digest_dict() for r in t.transitions],
                "loop": None if t.loop is None else t.loop.to_dict(),
            }
        )
        for t in templates
    ]
    audit["cross_split_duplicate_fingerprint_groups"] = (
        audit["cross_split_collision_count"] + legacy["cross_split_collision_count"]
    )
    if audit["cross_split_duplicate_fingerprint_groups"]:
        audit["status"] = "failed"
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(
            result,
            stream,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        stream.write("\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "cross_split_structural_collision_count",
                    "cross_split_semantic_collision_count",
                    "cross_split_duplicate_fingerprint_groups",
                )
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    raise SystemExit(0 if result["status"] == "passed" else 1)
