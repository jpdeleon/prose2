"""#55 B2: a worker process that dies (OOM kill, segfault in native code) must
fail the sequence promptly instead of leaving the parent waiting forever."""

import subprocess
import sys
import textwrap

# Generous bound: the fixed pool reports the dead worker in well under a
# second; before the fix the parent never returned at all.
_HANG_BOUND_S = 60

_SCRIPT = textwrap.dedent(
    """
    import os
    import numpy as np
    from prose import Block, Image
    from prose.core.sequence import SequenceParallel

    class DieOn(Block):
        def run(self, image):
            if image.metadata["id"] == 3:
                os._exit(1)

    seq = SequenceParallel([DieOn()], processes=2)
    try:
        seq.run(
            [Image(np.zeros((2, 2)), metadata={"id": i}) for i in range(8)],
            show_progress=False,
        )
    except RuntimeError as exc:
        print("RAISED:", exc)
        raise SystemExit(7)
    print("COMPLETED")
    """
)


def test_dead_worker_raises_instead_of_hanging(tmp_path):
    script = tmp_path / "dead_worker.py"
    script.write_text(_SCRIPT)
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=_HANG_BOUND_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            f"sequence still blocked {_HANG_BOUND_S}s after a worker died"
        ) from exc

    assert proc.returncode == 7, proc.stdout + proc.stderr
    assert "worker process died" in proc.stdout
