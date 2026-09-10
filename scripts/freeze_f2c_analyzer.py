"""仅建立/核验冻结元数据；不调用阶段构建、评分或物化入口。"""

import argparse
import json
from pathlib import Path

from keyed_gram.stage_f2c_freeze import create_freeze, verify_binding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("create", "verify"))
    parser.add_argument("--commit")
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--output", default="artifacts/stage_f2c_freeze")
    parser.add_argument("--evidence", default="artifacts/stage_f2c_compatibility")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.action == "create":
        result = create_freeze(
            root, args.commit, args.output, evidence_dir=args.evidence
        )
    else:
        result = verify_binding(
            root,
            args.output + "/freeze_binding.json",
            check_runtime=args.runtime,
        )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
