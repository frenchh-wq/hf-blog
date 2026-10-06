"""Claude research layer: market analysis, strategy generation and post-trade review.

Claude proposes; the backtester disposes. Nothing Claude suggests is traded until it beats the
incumbent strategy out-of-sample under the same risk rules.
"""
from __future__ import annotations

import json
import logging

import anthropic

from .strategies import PARAM_GRID, STRATEGY_DOCS, StrategySpec

log = logging.getLogger("agent.llm")

SYSTEM = """You are the research desk of an automated crypto trading system. You receive a market \
snapshot, backtest results and recent live/paper performance. Your job is to propose concrete, \
testable strategy configurations from a fixed menu of strategy templates, and to critique results \
honestly. Every proposal will be backtested walk-forward before use; proposals that overfit or \
don't beat the incumbent out-of-sample are discarded, so favour robust, simple parameter choices \
over curve-fitting. The system is long-only spot with a hard 2% risk per trade, 5% daily loss \
limit and 15% max drawdown; you cannot change those.

Strategy templates (name: behaviour; typical parameter values):
{templates}

Stops and take-profits are expressed in ATR multiples (stop_atr 1-4, take_profit_atr 1.5-10)."""

SCHEMA = {
    "type": "object",
    "properties": {
        "regime_assessment": {"type": "string"},
        "weaknesses": {"type": "array", "items": {"type": "string"}},
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": list(PARAM_GRID)},
                    "params": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"key": {"type": "string"}, "value": {"type": "number"}},
                            "required": ["key", "value"],
                            "additionalProperties": False,
                        },
                    },
                    "stop_atr": {"type": "number"},
                    "take_profit_atr": {"type": "number"},
                    "rationale": {"type": "string"},
                },
                "required": ["name", "params", "stop_atr", "take_profit_atr", "rationale"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["regime_assessment", "weaknesses", "proposals"],
    "additionalProperties": False,
}


class Researcher:
    def __init__(self, api_key: str, model: str = "claude-opus-5-5"):
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model
        templates = "\n".join(f"- {n}: {STRATEGY_DOCS[n]} params {PARAM_GRID[n]}" for n in PARAM_GRID)
        self.system = SYSTEM.format(templates=templates)

    def _ask(self, prompt: str, effort: str = "high") -> dict:
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=self.system,
            thinking={"type": "adaptive"},
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"Claude declined the request: {resp.stop_details}")
        if resp.stop_reason == "max_tokens":
            raise RuntimeError("Claude response truncated (max_tokens)")
        text = next(b.text for b in resp.content if b.type == "text")
        return json.loads(text)

    @staticmethod
    def _specs(data: dict) -> list[StrategySpec]:
        specs = []
        for p in data.get("proposals", []):
            allowed = PARAM_GRID[p["name"]]
            params = {kv["key"]: float(kv["value"]) for kv in p["params"] if kv["key"] in allowed}
            # fill anything Claude omitted with the middle of the grid
            for k, vals in allowed.items():
                params.setdefault(k, float(vals[len(vals) // 2]))
            specs.append(StrategySpec(
                name=p["name"], params=params, rationale=p["rationale"],
                stop_atr=min(max(p["stop_atr"], 0.5), 6.0),
                take_profit_atr=min(max(p["take_profit_atr"], 1.0), 15.0),
            ))
        return specs

    def research(self, market: dict, stats: dict, leaderboard: list[dict], incumbent: dict | None,
                 recent_perf: dict | None = None) -> tuple[dict, list[StrategySpec]]:
        prompt = (
            "Propose 3-5 strategy configurations for the next period.\n\n"
            f"<market_snapshot>\n{json.dumps(market, indent=1)}\n</market_snapshot>\n"
            f"<history_stats>\n{json.dumps(stats, indent=1)}\n</history_stats>\n"
            f"<grid_search_leaderboard description='top configs from a coarse grid; train = in-sample, test = out-of-sample'>\n"
            f"{json.dumps(leaderboard, indent=1, default=str)}\n</grid_search_leaderboard>\n"
            f"<incumbent>\n{json.dumps(incumbent, indent=1, default=str)}\n</incumbent>\n"
            f"<recent_performance>\n{json.dumps(recent_perf or {}, indent=1, default=str)}\n</recent_performance>\n"
            "Include at least one proposal that is not just a copy of the leaderboard."
        )
        data = self._ask(prompt)
        return data, self._specs(data)

    def review(self, report: dict) -> dict:
        """self_improvement: critique recent results and suggest adjustments."""
        prompt = (
            "Review this performance report. Identify weaknesses (losing regimes, stop placement, "
            "over-trading, signal lag) and propose up to 3 adjusted configurations that address them.\n\n"
            f"<report>\n{json.dumps(report, indent=1, default=str)}\n</report>"
        )
        data = self._ask(prompt, effort="medium")
        data["specs"] = self._specs(data)
        return data
