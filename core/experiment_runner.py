"""Think-Execute experiment runner.

Implements the three-phase loop:
  1. Perceive: read game state
  2. Think:    Jev recognizes situation → select script
              N.E.K.O personality layer shows expression + monologue
  3. Execute:  ScriptExecutor plays frame-precise input sequence
              Interrupt if state diverges from expectations
  4. Repeat

This replaces the per-frame DQN approach with a human-like
"recognize → think → execute" cycle.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import numpy as np

from .celeste_env import CelesteEnv
from .jev_agent import JevSituationAgent
from .script_library import ScriptLibrary
from .script_executor import ScriptExecutor
from .personality import PersonalityLayer, Expression
from .state_encoder import CelesteState, CelesteStateEncoder
from .dqn_agent import DQNAgent
from .reward_shaper import JevRewardShaper


class ThinkExecuteRunner:
    """Runs the think-execute loop for Celeste experiments."""

    def __init__(
        self,
        env: CelesteEnv,
        jev: JevSituationAgent | None,
        script_library: ScriptLibrary,
        dqn: DQNAgent | None = None,
        shaper: JevRewardShaper | None = None,
        personality: PersonalityLayer | None = None,
        logger=None,
    ):
        self.env = env
        self.jev = jev
        self.scripts = script_library
        self.dqn = dqn
        self.shaper = shaper
        self.personality = personality or PersonalityLayer(logger=logger)
        self.logger = logger
        self._encoder = CelesteStateEncoder()

        self._executor = ScriptExecutor(
            env=env,
            fps=env.fps,
            logger=logger,
            on_thought=self._on_executor_event,
            on_state_change=self._on_state_change,
        )

        self._running = False
        self._stop_flag = False
        self._episode_metrics: list[dict] = []

    async def run(self, episodes: int) -> dict[str, Any]:
        """Run the think-execute loop for N episodes."""
        self._running = True
        self._stop_flag = False

        for ep in range(episodes):
            if self._stop_flag:
                break

            metrics = await self._run_episode(ep)
            self._episode_metrics.append(metrics)

            if (ep + 1) % 10 == 0 and self.logger:
                self.logger.info(
                    "Episode %d: scripts=%d completed=%s thoughts=%d",
                    ep,
                    metrics["scripts_executed"],
                    metrics["completed"],
                    metrics["total_thoughts"],
                )

        summary = self._summarize()
        self._running = False
        return {"episodes": len(self._episode_metrics), "summary": summary}

    async def _run_episode(self, episode: int) -> dict[str, Any]:
        """Run one episode: perceive → think → execute loop."""
        self.personality.start_episode(episode)
        state = await self.env.reset()

        scripts_executed = 0
        scripts_succeeded = 0
        scripts_interrupted = 0
        total_thoughts = 0
        jev_calls = 0
        total_frames = 0
        start_time = time.perf_counter()
        completed = False

        while not self._stop_flag:
            # ── Phase 1: Perceive ────────────────────────────
            state_text = self._encoder.encode(state)

            # Check position memory
            memory_thought = self.personality.check_position_memory(
                state.pos_x, state.pos_y
            )

            # ── Phase 2: Think (Jev + N.E.K.O) ───────────────
            if self.jev:
                jev_result = await self.jev.recognize_situation(
                    state, self.scripts
                )
                jev_calls += 1
                situation = jev_result["situation"]
                script = self.scripts.get(situation)
                thought = await self.personality.on_thinking(
                    state_text, jev_result
                )
            else:
                script = self.scripts.get("simple_move_right")
                thought = "(no Jev) default: simple_move_right"
                jev_result = {"danger_prob": 0.5, "confidence_score": 2.0}

            if memory_thought:
                thought = f"{thought} {memory_thought}"

            total_thoughts += 1

            if self.logger:
                self.logger.info("[EP %d] Think: %s", episode, thought)

            if not script:
                script = self.scripts.get("simple_move_right")

            # ── Brief pause = "thinking time" (0.3-1.0s) ─────
            think_pause = 0.3 + min(0.7, jev_result.get("latency_ms", 300) / 1000.0)
            await asyncio.sleep(think_pause)

            # ── Phase 3: Execute frame-precise script ─────────
            exec_thought = await self.personality.on_executing(
                script.id, script.name, script.difficulty,
                jev_result.get("danger_prob", 0.5),
            )
            total_thoughts += 1

            result = await self._executor.execute(
                script,
                initial_state=state,
                allow_interrupt=True,
            )

            scripts_executed += 1
            total_frames += result["frames_executed"]

            if result["completed"]:
                scripts_succeeded += 1
                self.scripts.record_usage(script.id, True)
                await self.personality.on_success(script.id, script.name)
                total_thoughts += 1
            elif result.get("game_over"):
                await self.personality.on_failure(
                    script.id,
                    result.get("interrupt_reason", "game_over"),
                    position=(state.pos_x, state.pos_y),
                )
                total_thoughts += 1
                break
            elif result["interrupted"]:
                scripts_interrupted += 1
                self.scripts.record_usage(script.id, False)
                await self.personality.on_failure(
                    script.id,
                    result.get("interrupt_reason", "interrupted"),
                    position=(state.pos_x, state.pos_y),
                )
                total_thoughts += 1

            state = result["final_state"]

            # Check if level is complete
            if state.dist_to_goal < 5.0:
                completed = True
                self.set_expression(Expression.HAPPY)
                break

            # Safety: prevent infinite loops
            if scripts_executed > 50:
                if self.logger:
                    self.logger.warning("Episode %d: too many scripts, stopping", episode)
                break

        elapsed = time.perf_counter() - start_time
        self.personality.end_episode(completed)

        return {
            "episode": episode,
            "completed": completed,
            "scripts_executed": scripts_executed,
            "scripts_succeeded": scripts_succeeded,
            "scripts_interrupted": scripts_interrupted,
            "total_thoughts": total_thoughts,
            "jev_calls": jev_calls,
            "total_frames": total_frames,
            "elapsed_s": elapsed,
            "success_rate": scripts_succeeded / max(scripts_executed, 1),
        }

    async def _on_executor_event(self, text: str, context: dict):
        """Callback from ScriptExecutor for thought display."""
        if self.logger:
            self.logger.debug("[Executor] %s %s", text, context)

    async def _on_state_change(self, state: CelesteState):
        """Callback from ScriptExecutor for state observation."""
        pass

    def _summarize(self) -> dict[str, Any]:
        if not self._episode_metrics:
            return {"episodes": 0}

        completions = [m["completed"] for m in self._episode_metrics]
        scripts = [m["scripts_executed"] for m in self._episode_metrics]
        successes = [m["scripts_succeeded"] for m in self._episode_metrics]
        thoughts = [m["total_thoughts"] for m in self._episode_metrics]
        jev_calls = [m["jev_calls"] for m in self._episode_metrics]

        return {
            "episodes": len(self._episode_metrics),
            "completion_rate": sum(completions) / len(completions),
            "avg_scripts_per_episode": float(np.mean(scripts)),
            "avg_success_rate": float(np.mean([
                m["success_rate"] for m in self._episode_metrics
            ])),
            "avg_thoughts_per_episode": float(np.mean(thoughts)),
            "total_jev_calls": sum(jev_calls),
            "avg_jev_calls_per_episode": float(np.mean(jev_calls)),
            "script_stats": self.scripts.get_stats(),
            "personality_stats": self.personality.get_personality_stats(),
            "executor_stats": self._executor.get_stats(),
            "last_10_completion": sum(completions[-10:]) / max(len(completions[-10:]), 1),
        }

    def get_status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "expression": self.personality.expression.value,
            "recent_thoughts": self.personality.get_recent_thoughts(5),
            "env": self.env.get_stats(),
            "executor": self._executor.get_stats(),
            "personality": self.personality.get_personality_stats(),
        }

    def get_results(self, fmt: str = "summary") -> dict[str, Any]:
        if fmt == "full":
            return {
                "summary": self._summarize(),
                "episode_metrics": self._episode_metrics,
                "recent_thoughts": self.personality.get_recent_thoughts(20),
            }
        return self._summarize()

    async def stop(self):
        self._stop_flag = True
        self._running = False
        self._executor.request_interrupt()
