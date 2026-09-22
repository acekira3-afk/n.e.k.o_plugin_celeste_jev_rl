"""First data test: end-to-end mock experiment.

Simulates the Think-Execute loop with a mock Celeste environment
and mock Jev agent to validate the full architecture and generate
the first dataset of metrics, thoughts, and personality states.

Run: python3 tests/test_first_data.py
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import os
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.state_encoder import CelesteState, CelesteStateEncoder
from core.script_library import ScriptLibrary
from core.script_executor import ScriptExecutor
from core.personality import PersonalityLayer, Expression
from core.dqn_agent import DQNAgent
from core.experiment_runner import ThinkExecuteRunner
from core.celeste_env import CelesteEnv


class MockCelesteEnv:
    """Simulated Celeste environment for first data test.

    Models a simple level: start at x=0, goal at x=500.
    Each script moves the player forward some amount.
    Random events (spikes, falls) can interrupt scripts.
    """

    def __init__(self, fps: int = 60, seed: int = 42):
        self.fps = fps
        self._rng = random.Random(seed)
        self._pos_x = 0.0
        self._pos_y = 0.0
        self._vel_x = 0.0
        self._vel_y = 0.0
        self._stamina = 1.0
        self._has_dash = True
        self._on_ground = True
        self._climbing = False
        self._dist_to_goal = 500.0
        self._step_count = 0
        self._episode = 0
        self._obstacle_distances = [100.0] * 8
        self._obstacle_types = [0] * 16
        self._dead = False

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
        self._dist_to_goal = 500.0
        self._dead = False
        self._obstacle_distances = [80.0, 60.0, 40.0, 100.0, 200.0, 200.0, 30.0, 50.0]
        self._obstacle_types = [2, 1, 3, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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

        self._dist_to_goal = max(0.0, 500.0 - self._pos_x)

        if self._rng.random() < 0.03 and self._step_count > 5:
            self._dead = True
            return self._get_state(), -1.0, True, {"death": "spike"}

        reward = progress / 500.0
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
            "target": "mock_celeste",
            "fps": self.fps,
        }


class MockJevAgent:
    """Mock Jev agent that picks scripts based on simple heuristics."""

    def __init__(self, logger=None):
        self.logger = logger
        self._call_count = 0
        self._total_latency = 0.0
        self._encoder = CelesteStateEncoder()
        self._rng = random.Random(123)

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

    async def evaluate_action(self, state, action_name):
        self._call_count += 1
        danger = 0.3
        quality = 3.0
        return {
            "danger_prob": danger,
            "quality_score": quality,
            "shaped_reward": 0.0,
        }

    def get_stats(self):
        avg = (self._total_latency / self._call_count) if self._call_count else 0
        return {
            "total_calls": self._call_count,
            "avg_latency_ms": avg,
            "est_cost_usd": self._call_count * 0.000194,
        }

    async def close(self):
        pass


async def run_first_data_test():
    """Run the first data test and generate a report."""
    print("=" * 60)
    print("  Celeste Jev RL — First Data Test")
    print("  Think-Execute Architecture (Mock)")
    print("=" * 60)
    print()

    env = MockCelesteEnv(fps=60, seed=42)
    scripts = ScriptLibrary()
    jev = MockJevAgent()
    personality = PersonalityLayer()

    from core.experiment_runner import ThinkExecuteRunner
    runner = ThinkExecuteRunner(
        env=env,
        jev=jev,
        script_library=scripts,
        personality=personality,
        logger=None,
    )

    episodes = 50
    print(f"Running {episodes} episodes...")
    print()

    t0 = time.perf_counter()
    result = await runner.run(episodes=episodes)
    elapsed = time.perf_counter() - t0

    summary = result["summary"]

    print()
    print("-" * 60)
    print("  RESULTS SUMMARY")
    print("-" * 60)
    print(f"  Episodes:             {summary['episodes']}")
    print(f"  Completion rate:      {summary['completion_rate']:.1%}")
    print(f"  Avg scripts/episode:  {summary['avg_scripts_per_episode']:.1f}")
    print(f"  Avg script success:   {summary['avg_success_rate']:.1%}")
    print(f"  Avg thoughts/episode:{summary['avg_thoughts_per_episode']:.1f}")
    print(f"  Total Jev calls:      {summary['total_jev_calls']}")
    print(f"  Avg Jev calls/ep:     {summary['avg_jev_calls_per_episode']:.1f}")
    print(f"  Last 10 completion:   {summary['last_10_completion']:.1%}")
    print(f"  Elapsed:              {elapsed:.1f}s")
    print()

    print("-" * 60)
    print("  SCRIPT USAGE STATS")
    print("-" * 60)
    script_stats = summary["script_stats"]
    for sid, stats in sorted(script_stats.items(), key=lambda x: -x[1]["uses"]):
        if stats["uses"] > 0:
            rate = stats["success_rate"]
            print(f"  {sid:30s} uses={stats['uses']:3d}  success={rate:.0%}")
    print()

    print("-" * 60)
    print("  PERSONALITY STATS")
    print("-" * 60)
    pstats = summary["personality_stats"]
    print(f"  Final expression:     {pstats['current_expression']}")
    print(f"  Total deaths:        {pstats['total_deaths']}")
    print(f"  Total successes:     {pstats['total_successes']}")
    print(f"  Position memories:   {pstats['position_memories']}")
    print(f"  Total thoughts:      {pstats['total_thoughts']}")
    print()

    print("-" * 60)
    print("  EXECUTOR STATS")
    print("-" * 60)
    estats = summary["executor_stats"]
    print(f"  Scripts run:         {estats['total_scripts_run']}")
    print(f"  Frames executed:    {estats['total_frames_executed']}")
    print(f"  Interrupts:         {estats['total_interrupts']}")
    print(f"  Interrupt rate:      {estats['interrupt_rate']:.1%}")
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

    print("-" * 60)
    print("  EPISODE-BY-EPISODE (first 10 + last 5)")
    print("-" * 60)
    metrics = result.get("episode_metrics", runner._episode_metrics)
    for m in metrics[:10]:
        print(f"  EP {m['episode']:3d}: scripts={m['scripts_executed']:2d} "
              f"ok={m['scripts_succeeded']:2d} "
              f"interrupts={m['scripts_interrupted']:2d} "
              f"thoughts={m['total_thoughts']:2d} "
              f"done={'Y' if m['completed'] else 'N'} "
              f"jev={m['jev_calls']:2d} "
              f"frames={m['total_frames']:3d} "
              f"t={m['elapsed_s']:.1f}s")
    if len(metrics) > 15:
        print("  ...")
        for m in metrics[-5:]:
            print(f"  EP {m['episode']:3d}: scripts={m['scripts_executed']:2d} "
                  f"ok={m['scripts_succeeded']:2d} "
                  f"interrupts={m['scripts_interrupted']:2d} "
                  f"thoughts={m['total_thoughts']:2d} "
                  f"done={'Y' if m['completed'] else 'N'} "
                  f"jev={m['jev_calls']:2d} "
                  f"frames={m['total_frames']:3d} "
                  f"t={m['elapsed_s']:.1f}s")
    print()

    report = {
        "test_name": "first_data_test",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "episodes": episodes,
        "elapsed_seconds": round(elapsed, 2),
        "summary": summary,
        "episode_metrics": metrics,
        "recent_thoughts": thoughts,
    }

    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs",
        "first_data_test_report.json",
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"  Report saved: {output_path}")
    print()
    print("=" * 60)
    print("  First data test complete.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_first_data_test())
