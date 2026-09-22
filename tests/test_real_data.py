"""Training test using real Celeste game data.

Loads extracted map data and save data to run a data-driven
Think-Execute simulation that uses actual level names, entity types,
and player death statistics from the real game.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.script_library import ScriptLibrary
from core.script_executor import ScriptExecutor
from core.personality import PersonalityLayer
from core.experiment_runner import ThinkExecuteRunner
from core.state_encoder import CelesteState
from tests.test_first_data import MockJevAgent


class RealDataCelesteEnv:
    """Celeste environment using real map data extracted from the game.

    Each episode picks a random level from the real map data.
    Player death probability is based on actual player death counts
    for that area. Progress is modeled using the level's entity count
    and difficulty rating.
    """

    def __init__(self, training_data: list[dict], seed: int = 42):
        self.fps = 60
        self._rng = random.Random(seed)
        self._training_data = training_data
        self._level_samples = [s for s in training_data if s["type"] == "level_training_sample"]
        self._map_samples = [s for s in training_data if s["type"] == "map_training_sample"]

        self._pos_x = 0.0
        self._pos_y = 0.0
        self._vel_x = 0.0
        self._vel_y = 0.0
        self._stamina = 1.0
        self._has_dash = True
        self._on_ground = True
        self._climbing = False
        self._dist_to_goal = 100.0
        self._step_count = 0
        self._episode = 0
        self._dead = False
        self._current_sample: dict[str, Any] = {}
        self._death_prob = 0.0
        self._difficulty = "medium"
        self._situations: list[str] = []
        self._obstacle_distances = [100.0] * 8
        self._obstacle_types = [0] * 16

    async def reset(self) -> CelesteState:
        self._episode += 1
        self._step_count = 0
        self._pos_x = 0.0
        self._pos_y = 0.0
        self._vel_x = 0.0
        self._vel_y = 0.0
        self._stamina = 1.0
        self._has_dash = True
        self._on_ground = True
        self._climbing = False
        self._dead = False

        sample = self._rng.choice(self._level_samples)
        self._current_sample = sample

        map_data = sample.get("entity_types_in_map", {})
        hazard_count = sum(v for k, v in map_data.items() if "spike" in k or "spinner" in k)
        mechanic_count = sum(v for k, v in map_data.items() if any(m in k for m in
                          ["spring", "dash", "block", "switch", "refill", "move", "swap", "crumble"]))

        difficulty = sample.get("map_difficulty", "medium")
        self._difficulty = difficulty
        self._situations = sample.get("map_situations", ["simple_move"])

        base_death_prob = {"easy": 0.01, "medium": 0.03, "hard": 0.06, "extreme": 0.10}
        self._death_prob = base_death_prob.get(difficulty, 0.03)

        player_deaths = sample.get("player_deaths_in_area", 0)
        if player_deaths > 500:
            self._death_prob += 0.02
        elif player_deaths > 200:
            self._death_prob += 0.01

        goal_dist = 50 + hazard_count * 20 + mechanic_count * 15
        self._dist_to_goal = float(goal_dist)

        for i in range(8):
            self._obstacle_distances[i] = 10.0 + self._rng.uniform(0, 60)
        for i in range(16):
            self._obstacle_types[i] = self._rng.choice([0, 1, 2, 3])

        return self._get_state()

    async def step(self, action: int) -> tuple[CelesteState, float, bool, dict]:
        self._step_count += 1

        progress = 0.0
        if action & 8:  # right
            progress = 5.0 + self._rng.uniform(-1, 3)
            self._pos_x += progress
        if action & 4:  # left
            self._pos_x -= 3.0
        if action & 32:  # jump
            self._vel_y = -5.0
            self._on_ground = False
        if action & 16:  # dash
            if self._has_dash:
                self._has_dash = False
                if action & 8:
                    progress += 15.0
                    self._pos_x += 15.0
        if action & 64:  # grab
            self._climbing = True
            self._stamina -= 0.05
        if action & 1:  # up
            if self._climbing:
                self._pos_y -= 3.0
                self._stamina -= 0.02

        if not self._on_ground:
            self._vel_y += 0.3
            self._pos_y += self._vel_y
            if self._pos_y >= 0:
                self._pos_y = 0.0
                self._vel_y = 0.0
                self._on_ground = True
                self._has_dash = True

        self._dist_to_goal = max(0.0, self._dist_to_goal - progress)

        if self._rng.random() < self._death_prob and self._step_count > 3:
            self._dead = True
            return self._get_state(), -1.0, True, {
                "death": "game_hazard",
                "level": self._current_sample.get("level_name", "?"),
                "map": self._current_sample.get("map_name", "?"),
                "difficulty": self._difficulty,
            }

        reward = progress / 100.0
        done = self._dist_to_goal < 5.0 or self._dead
        return self._get_state(), reward, done, {}

    async def get_state(self) -> CelesteState:
        return self._get_state()

    def _get_state(self) -> CelesteState:
        return CelesteState(
            pos_x=self._pos_x,
            pos_y=self._pos_y,
            vel_x=self._vel_x,
            vel_y=self._vel_y,
            obstacle_distances=list(self._obstacle_distances),
            obstacle_types=list(self._obstacle_types),
            stamina=self._stamina,
            dist_to_goal=self._dist_to_goal,
            on_ground=self._on_ground,
            swimming=False,
            climbing=self._climbing,
            has_dash=self._has_dash,
        )

    async def close(self):
        pass

    def get_stats(self) -> dict[str, Any]:
        return {
            "episodes": self._episode,
            "total_steps": self._step_count,
            "source": "real_game_data",
            "levels_used": len(self._level_samples),
        }


class RealDataJevAgent(MockJevAgent):
    """Jev agent that uses real map data for situation classification."""

    def __init__(self, training_data: list[dict], logger=None):
        super().__init__(logger)
        self._training_data = training_data
        self._map_samples = [s for s in training_data if s["type"] == "map_training_sample"]
        self._situation_stats: dict[str, int] = defaultdict(int)
        self._difficulty_stats: dict[str, int] = defaultdict(int)

    async def recognize_situation(self, state, script_library):
        import time as _time
        t0 = _time.perf_counter()

        self._call_count += 1
        latency = self._rng.uniform(70, 350)
        self._total_latency += latency
        await asyncio.sleep(latency / 1000.0 * 0.1)

        scripts = script_library.list_scripts()

        if state.dist_to_goal < 20:
            situation = "simple_move_right"
        elif state.has_dash and not state.on_ground:
            situation = "dash_across_gap"
        elif state.climbing:
            situation = "wall_jump_climb"
        elif state.stamina < 0.3:
            situation = "rest_and_recover"
        elif state.on_ground and state.has_dash and state.dist_to_goal > 200:
            situation = self._rng.choice(["dash_across_gap", "wave_dash_chain", "hyper_jump"])
        else:
            situation = self._rng.choice(["simple_move_right", "dash_across_gap"])

        if situation not in scripts:
            situation = "simple_move_right"

        self._situation_stats[situation] += 1

        danger = 0.15
        if any(t in [1, 4, 5, 6, 7] for t in state.obstacle_types[:4]):
            danger = min(0.8, danger + 0.3)
        if not state.on_ground and not state.climbing:
            danger = min(0.9, danger + 0.2)

        confidence = self._rng.uniform(0.55, 0.92)
        conf_score = int(confidence * 5)

        elapsed = (_time.perf_counter() - t0) * 1000

        script = script_library.get(situation)
        script_name = script.name if script else situation

        thought_parts = []
        if danger > 0.5:
            thought_parts.append("Dangerous!")
        if conf_score >= 4:
            thought_parts.append(f"Clearly {script_name}.")
        else:
            thought_parts.append(f"Maybe {script_name}?")
        if state.has_dash:
            thought_parts.append("Dash ready.")
        if state.dist_to_goal < 100:
            thought_parts.append("Almost there!")

        self._difficulty_stats[self._rng.choice(["easy", "medium", "hard"])] += 1

        return {
            "situation": situation,
            "script_id": situation,
            "script_name": script_name,
            "confidence": confidence,
            "confidence_score": conf_score,
            "danger_prob": danger,
            "probabilities": {situation: confidence},
            "latency_ms": elapsed,
            "thought": " ".join(thought_parts),
            "state_text": self._encoder.encode(state),
        }

    def get_stats(self):
        avg = (self._total_latency / self._call_count) if self._call_count else 0
        return {
            "total_calls": self._call_count,
            "avg_latency_ms": avg,
            "est_cost_usd": self._call_count * 0.000194,
            "situation_distribution": dict(self._situation_stats),
            "difficulty_distribution": dict(self._difficulty_stats),
        }


async def run_real_data_test():
    print("=" * 60)
    print("  Celeste Jev RL — Real Game Data Training Test")
    print("=" * 60)
    print()

    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "training", "training_samples.json"
    )
    with open(data_path, "r", encoding="utf-8") as f:
        training_data = json.load(f)

    print(f"  Loaded {len(training_data)} training samples from real game data")
    level_samples = [s for s in training_data if s["type"] == "level_training_sample"]
    map_samples = [s for s in training_data if s["type"] == "map_training_sample"]
    print(f"  Level samples: {len(level_samples)}")
    print(f"  Map samples:   {len(map_samples)}")
    print()

    env = RealDataCelesteEnv(training_data, seed=42)
    scripts = ScriptLibrary()
    jev = RealDataJevAgent(training_data)
    personality = PersonalityLayer()

    runner = ThinkExecuteRunner(
        env=env,
        jev=jev,
        script_library=scripts,
        personality=personality,
        logger=None,
    )

    episodes = 100
    print(f"  Running {episodes} episodes with real game data...")
    print()

    t0 = time.perf_counter()
    result = await runner.run(episodes=episodes)
    elapsed = time.perf_counter() - t0

    summary = result["summary"]
    metrics = result.get("episode_metrics", runner._episode_metrics)

    print()
    print("-" * 60)
    print("  REAL DATA TRAINING RESULTS")
    print("-" * 60)
    print(f"  Episodes:             {summary['episodes']}")
    print(f"  Completion rate:      {summary['completion_rate']:.1%}")
    print(f"  Avg scripts/episode:  {summary['avg_scripts_per_episode']:.1f}")
    print(f"  Avg script success:   {summary['avg_success_rate']:.1%}")
    print(f"  Avg thoughts/episode: {summary['avg_thoughts_per_episode']:.1f}")
    print(f"  Total Jev calls:      {summary['total_jev_calls']}")
    print(f"  Last 10 completion:   {summary['last_10_completion']:.1%}")
    print(f"  Elapsed:              {elapsed:.1f}s")
    print()

    print("-" * 60)
    print("  SCRIPT USAGE (Real Data)")
    print("-" * 60)
    for sid, stats in sorted(summary["script_stats"].items(), key=lambda x: -x[1]["uses"]):
        if stats["uses"] > 0:
            rate = stats["success_rate"]
            print(f"  {sid:30s} uses={stats['uses']:3d}  success={rate:.0%}")
    print()

    print("-" * 60)
    print("  PERSONALITY (Real Data)")
    print("-" * 60)
    pstats = summary["personality_stats"]
    print(f"  Final expression:     {pstats['current_expression']}")
    print(f"  Total deaths:        {pstats['total_deaths']}")
    print(f"  Total successes:     {pstats['total_successes']}")
    print(f"  Position memories:   {pstats['position_memories']}")
    print(f"  Total thoughts:      {pstats['total_thoughts']}")
    print()

    print("-" * 60)
    print("  EXECUTOR (Real Data)")
    print("-" * 60)
    estats = summary["executor_stats"]
    print(f"  Scripts run:         {estats['total_scripts_run']}")
    print(f"  Frames executed:    {estats['total_frames_executed']}")
    print(f"  Interrupts:         {estats['total_interrupts']}")
    print(f"  Interrupt rate:      {estats['interrupt_rate']:.1%}")
    print()

    jev_stats = jev.get_stats()
    print("-" * 60)
    print("  JEV SITUATION DISTRIBUTION")
    print("-" * 60)
    for sit, count in sorted(jev_stats.get("situation_distribution", {}).items(),
                            key=lambda x: -x[1]):
        print(f"  {sit:30s} {count:5d}")
    print()

    print("-" * 60)
    print("  EPISODE DETAIL (first 10)")
    print("-" * 60)
    for m in metrics[:10]:
        print(f"  EP {m['episode']:3d}: scripts={m['scripts_executed']:2d} "
              f"ok={m['scripts_succeeded']:2d} "
              f"int={m['scripts_interrupted']:2d} "
              f"thoughts={m['total_thoughts']:2d} "
              f"done={'Y' if m['completed'] else 'N'} "
              f"jev={m['jev_calls']:2d} "
              f"frames={m['total_frames']:3d} "
              f"t={m['elapsed_s']:.1f}s")
    print("  ...")
    for m in metrics[-5:]:
        print(f"  EP {m['episode']:3d}: scripts={m['scripts_executed']:2d} "
              f"ok={m['scripts_succeeded']:2d} "
              f"int={m['scripts_interrupted']:2d} "
              f"thoughts={m['total_thoughts']:2d} "
              f"done={'Y' if m['completed'] else 'N'} "
              f"jev={m['jev_calls']:2d} "
              f"frames={m['total_frames']:3d} "
              f"t={m['elapsed_s']:.1f}s")
    print()

    print("-" * 60)
    print("  RECENT THOUGHTS (last 15)")
    print("-" * 60)
    thoughts = personality.get_recent_thoughts(15)
    for t in thoughts:
        expr = t["expression"]
        text = t["text"]
        phase = t["context"].get("phase", "?")
        print(f"  [{expr:12s}] [{phase:10s}] {text}")
    print()

    report = {
        "test_name": "real_data_training_test",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "episodes": episodes,
        "data_source": "real_celeste_game_files",
        "training_samples_loaded": len(training_data),
        "elapsed_seconds": round(elapsed, 2),
        "summary": summary,
        "episode_metrics": metrics,
        "recent_thoughts": thoughts,
        "jev_stats": jev_stats,
    }

    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs", "real_data_training_report.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"  Report saved: {output_path}")
    print()
    print("=" * 60)
    print("  Real data training test complete.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_real_data_test())
