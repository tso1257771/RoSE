"""The benchmark's shared-event-loop wrapper around SeisBench ``classify``."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase_picking" / "benchmark"))

import _classify  # noqa: E402


class _AsyncModel:
    def __init__(self):
        self.loops = []

    async def classify_async(self, stream, **kwargs):
        self.loops.append(asyncio.get_running_loop())
        return ("picks", stream, kwargs)


class _OldModel:
    def classify(self, stream, **kwargs):
        return ("old", stream, kwargs)


def test_same_loop_for_every_call():
    model = _AsyncModel()
    out = _classify.classify(model, "s1", P_threshold=0.1)
    assert out == ("picks", "s1", {"P_threshold": 0.1})
    _classify.classify(model, "s2")
    assert model.loops[0] is model.loops[1]
    assert not model.loops[0].is_closed()


def test_falls_back_to_classify_without_async():
    assert _classify.classify(_OldModel(), "s", S_threshold=0.2) == ("old", "s", {"S_threshold": 0.2})
