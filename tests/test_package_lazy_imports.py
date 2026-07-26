from __future__ import annotations

import subprocess
import sys


def test_package_import_does_not_eagerly_load_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import keyed_gram; "
                "assert 'torch' not in sys.modules; "
                "assert keyed_gram.GramModelConfig is not None; "
                "assert 'torch' not in sys.modules; "
                "assert keyed_gram.GramTransformer is not None; "
                "assert 'torch' in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
