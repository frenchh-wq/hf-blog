import json
from types import SimpleNamespace

from agent.llm import Researcher


class FakeMessages:
    def __init__(self, payload):
        self.payload, self.kwargs = payload, None

    def create(self, **kw):
        self.kwargs = kw
        return SimpleNamespace(stop_reason="end_turn", stop_details=None,
                               content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=json.dumps(self.payload))])


def test_research_parses_and_sanitises_proposals():
    payload = {
        "regime_assessment": "choppy range",
        "weaknesses": [],
        "proposals": [
            {"name": "rsi_reversion", "params": [{"key": "lower", "value": 28}, {"key": "bogus", "value": 1}],
             "stop_atr": 50, "take_profit_atr": 4, "rationale": "fade dips"},
        ],
    }
    r = Researcher("test-key")
    fake = FakeMessages(payload)
    r.client = SimpleNamespace(beta=SimpleNamespace(messages=fake))
    data, specs = r.research({"price": 1}, {}, [], None)
    assert data["regime_assessment"] == "choppy range"
    s = specs[0]
    assert s.params["lower"] == 28 and "bogus" not in s.params
    assert set(s.params) == {"period", "lower", "upper", "trend"}  # missing params filled
    assert s.stop_atr == 6.0  # clamped
    assert fake.kwargs["model"] == "claude-opus-5-5"
    assert fake.kwargs["output_config"]["format"]["type"] == "json_schema"
    assert fake.kwargs["thinking"] == {"type": "adaptive"}
