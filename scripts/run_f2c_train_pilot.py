#!/usr/bin/env python3
"""只运行未冻结的 train 预实验；绝不调用正式 materialize/audit 入口。"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from keyed_gram.authsynth_symbolic_shared import GrammarLimits
from keyed_gram.authsynth_symbolic_verifier.benchmark import (
    generate_symbolic_smoke_cases,
)
from keyed_gram.authsynth_symbolic_verifier.evaluator import METHODS, evaluate_cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    # 先排他创建运行目录；普通预实验也不覆盖已有结果。
    args.output.mkdir(parents=True, exist_ok=False)
    cases = generate_symbolic_smoke_cases(
        "container-train-pilot-v1",
        grammar=GrammarLimits(4, 5, 16, True),
        replay_budget=16,
        solver_timeout_ms=3000,
    )
    assert {case.split for case in cases} == {"train", "calibration"}
    train = tuple(case for case in cases if case.split == "train")
    assert len(train) == 40
    metrics = []
    for method in METHODS:
        print(f"开始 train 方法：{method.value}", flush=True)
        bundle = evaluate_cases(
            train, replay_budget=16, solver_timeout_ms=3000, methods=(method,)
        )
        result = dict(bundle["metrics"][0])
        metrics.append(result)
        with (args.output / f"method_{len(metrics):02d}.json").open(
            "x", encoding="utf-8"
        ) as handle:
            json.dump(
                result, handle, ensure_ascii=False, sort_keys=True, allow_nan=False
            )
            handle.write("\n")
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    summary = {
        "schema_version": 2,
        "patch_metric_semantics": "location_component_atoms_v1",
        "run_kind": "nonformal_train_only_pilot",
        "case_count": len(train),
        "domain_counts": dict(Counter(case.domain for case in train)),
        "evaluated_splits": ["train"],
        "method_metrics": metrics,
        "settings": {
            "replay_budget": 16,
            "solver_timeout_ms": 3000,
            "max_disjuncts": 4,
            "max_literals_per_conjunction": 5,
            "max_total_literals": 16,
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metric_limitations": [
            "atom 按效果签名/字段组件定位，不代表合成公式本身已正确修复",
            "Safe Utility 与 MPR 在当前 unary evaluator 中使用相同计数",
            "指标仍属待审查草稿；train 结果不支持 unseen-family 泛化结论",
        ],
        "formal_calibration_started": False,
        "formal_development_started": False,
        "formal_locked_started": False,
        "analyzer_frozen": False,
        "ready_for_formal_f2c_audit": False,
        "ready_for_stage_f2d_relational_contracts": False,
        "private_data_used": False,
    }
    with (args.output / "train_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
