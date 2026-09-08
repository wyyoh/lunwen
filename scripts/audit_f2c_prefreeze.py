"""只审计 train/calibration 非正式 fixture；不调用求解器或正式入口。"""

import json

from keyed_gram.authsynth_symbolic_shared import GrammarLimits
from keyed_gram.authsynth_symbolic_verifier.benchmark import (
    collision_audit,
    generate_symbolic_smoke_cases,
)


def main():
    cases = generate_symbolic_smoke_cases(
        "prefreeze-content-isolation-review",
        grammar=GrammarLimits(4, 5, 16, True),
        replay_budget=16,
        solver_timeout_ms=3000,
    )
    assert {case.split for case in cases} == {"train", "calibration"}
    collision = collision_audit(cases)
    result = {
        "schema_version": 1,
        "audit_kind": "nonformal_prefreeze_content_isolation",
        "status": "failed" if collision["cross_split_collision_count"] else "passed",
        "fixture_case_count": len(cases),
        "collision_audit": collision,
        "analyzer_frozen": False,
        "formal_benchmark_materialized": False,
        "development_scored": False,
        "locked_test_scored": False,
        "solver_executed": False,
        "readiness_for_formal_audit": False,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
