"""SEN-109 (reopened): test fixtures must be stable across PYTHONHASHSEED.

Builtin ``hash(str)`` is randomized per interpreter, so tests that relied on
``hash(label)`` for a seed produced different frames on different runs. The
fixtures now use ``zlib.crc32``, which is deterministic. This test spawns two
subprocesses with different PYTHONHASHSEED values and confirms the generated
train.parquet is byte-identical by DataFrame comparison.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

GEN_SCRIPT = r"""
import sys
from pathlib import Path
sys.path.insert(0, r"{repo}")
from tests.fixtures import make_cicids_frame, write_splits, CANONICAL_12
out = Path(sys.argv[1])
write_splits(out, {{c: 20 for c in CANONICAL_12}}, seed=0)
"""


def _run_with_hashseed(tmp: Path, seed_val: str) -> pd.DataFrame:
    env = {**os.environ, "PYTHONHASHSEED": seed_val}
    script = GEN_SCRIPT.format(repo=str(REPO_ROOT))
    subprocess.run(
        [sys.executable, "-c", script, str(tmp)],
        env=env, check=True, capture_output=True,
    )
    return pd.read_parquet(tmp / "train.parquet")


def test_fixtures_identical_across_pythonhashseed(tmp_path):
    a = _run_with_hashseed(tmp_path / "a", "1")
    b = _run_with_hashseed(tmp_path / "b", "2")
    pd.testing.assert_frame_equal(a, b)
