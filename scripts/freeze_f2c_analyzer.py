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
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.action == "create":
        result = create_freeze(root, args.commit, "artifacts/stage_f2c_freeze")
    else:
        result = verify_binding(
            root,
            "artifacts/stage_f2c_freeze/freeze_binding.json",
            check_runtime=args.runtime,
        )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
