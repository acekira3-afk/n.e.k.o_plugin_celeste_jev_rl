"""Jev reward shaper for DQN training.

Uses Jev's danger probability (noul) and action quality (score)
to shape the reward signal during DQN training.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .state_encoder import CelesteState, CelesteStateEncoder
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .jev_agent import JevSituationAgent


class JevRewardShaper:
    """Shapes DQN rewards using Jev's semantic judgments."""

    def __init__(self, jev: "JevSituationAgent", logger=None):
        self.jev = jev
        self.logger = logger
        self._encoder = CelesteStateEncoder()
        self._cache: dict[str, dict] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    async def shape(
        self,
        state: CelesteState | list,
        action_idx: int,
        action_name: str,
        base_reward: float,
    ) -> float:
        """Return shaped reward = base_reward + jev_bonus."""
        if isinstance(state, list):
            state = CelesteState.from_array(state)

        cache_key = self._cache_key(state, action_name)
        if cache_key in self._cache:
            self._cache_hits += 1
            eval_result = self._cache[cache_key]
        else:
            self._cache_misses += 1
            eval_result = await self.jev.evaluate_action(state, action_name)
            if len(self._cache) > 500:
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = eval_result

        danger = eval_result.get("danger_prob", 0.5)
        quality = eval_result.get("quality_score", 2.0)
        shaped = eval_result.get("shaped_reward", 0.0)

        penalty = 0.0
        if danger > 0.7:
            penalty = -0.5 * (danger - 0.7) / 0.3

        bonus = 0.0
        if quality > 3.0:
            bonus = 0.2 * (quality - 3.0) / 2.0

        return base_reward + penalty + bonus

    async def batch_shape(
        self,
        transitions: list[tuple[CelesteState, int, str, float]],
    ) -> list[float]:
        """Shape rewards for a batch of transitions."""
        tasks = [
            self.shape(state, action_idx, action_name, base_reward)
            for state, action_idx, action_name, base_reward in transitions
        ]
        return await asyncio.gather(*tasks)

    def _cache_key(self, state: CelesteState, action: str) -> str:
        return (
            f"{state.pos_x:.0f},{state.pos_y:.0f},"
            f"{state.vel_x:.0f},{state.vel_y:.0f},"
            f"{state.has_dash},{state.on_ground},{action}"
        )

    def get_stats(self) -> dict[str, Any]:
        total = self._cache_hits + self._cache_misses
        jev_stats = self.jev.get_stats()
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": self._cache_hits / total if total else 0,
            "cache_size": len(self._cache),
            "jev_calls": jev_stats["total_calls"],
            "jev_avg_latency_ms": jev_stats["avg_latency_ms"],
            "jev_est_cost_usd": jev_stats["est_cost_usd"],
        }

    def clear_cache(self):
        self._cache.clear()
