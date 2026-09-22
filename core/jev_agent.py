"""Jev situation-recognition agent.

Instead of choosing per-frame actions, Jev identifies the current
gameplay situation and selects a frame-precise script to execute.
This turns Jev's 70-500ms latency from a liability into a feature:
it becomes the "thinking time" before a human-like scripted sequence.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from .state_encoder import CelesteState, CelesteStateEncoder
from .script_library import ScriptLibrary


class JevSituationAgent:
    """Jev agent that recognizes situations and picks scripts.

    Decision loop:
      1. Encode game state → text
      2. Ask Jev: "What situation is this?" (Choice)
      3. Ask Jev: "How confident?" (Score)
      4. Ask Jev: "Is this dangerous?" (Noul)
      5. Return situation_id + confidence + danger_prob

    The caller (ScriptExecutor) then plays the matching script.
    """

    SITUATION_QUESTION = {
        "type": "choice",
        "instructions": (
            "You are playing Celeste, a precision platformer. "
            "Given the current game state, identify the gameplay situation. "
            "Consider: player position, velocity, nearby obstacles (spikes, walls, platforms), "
            "available mechanics (dash, climb, jump), stamina, and distance to goal. "
            "Select the situation that best describes what the player should do next."
        ),
    }

    DANGER_QUESTION = {
        "type": "noul",
        "instructions": (
            "Is the current state immediately dangerous? "
            "(about to hit spikes, falling into a pit, or stuck with no dash while climbing)"
        ),
    }

    CONFIDENCE_QUESTION = {
        "type": "score",
        "instructions": "How confident are you in this situation assessment?",
        "criteria": [
            "very_unsure",
            "unsure",
            "neutral",
            "confident",
            "very_confident",
        ],
    }

    def __init__(
        self,
        api_key: str,
        endpoint: str = "https://api.typesafe.ai/v1/systemone",
        model: str = "jev-latest",
        logger=None,
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model
        self.logger = logger
        self._client: httpx.AsyncClient | None = None
        self._call_count = 0
        self._total_latency = 0.0
        self._encoder = CelesteStateEncoder()

    async def _ensure_client(self):
        if not self._client:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, connect=5.0),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )

    async def recognize_situation(
        self,
        state: CelesteState | dict | list,
        script_library: ScriptLibrary,
    ) -> dict[str, Any]:
        """Ask Jev to recognize the current situation and pick a script.

        Returns:
            {
                "situation": "dash_across_gap",
                "script_id": "dash_across_gap",
                "confidence": 0.82,
                "danger_prob": 0.15,
                "latency_ms": 312,
                "thought": "Gap ahead with dash available → dash_across_gap",
            }
        """
        await self._ensure_client()

        state_text = self._encoder.encode(state)
        options = script_library.get_options_for_jev()

        questions = {
            "situation": {
                **self.SITUATION_QUESTION,
                "criteria": options,
            },
            "danger": self.DANGER_QUESTION,
            "confidence": self.CONFENCE_QUESTION_ALT() if False else self.CONFIDENCE_QUESTION,
        }

        payload = {
            "model": self.model,
            "state": state_text,
            "questions": questions,
        }

        t0 = time.perf_counter()
        try:
            resp = await self._client.post(self.endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            if self.logger:
                self.logger.error("Jev API error: %s", e.response.status_code)
            return self._fallback(state, script_library, str(e))
        except Exception as e:
            if self.logger:
                self.logger.error("Jev request failed: %s", e)
            return self._fallback(state, script_library, str(e))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._call_count += 1
        self._total_latency += elapsed_ms

        answers = data.get("answers", {})

        sit_answer = answers.get("situation", {})
        situation_id = sit_answer.get("choice", "simple_move_right")
        probabilities = sit_answer.get("probabilities", {})
        confidence_val = sit_answer.get("confidence", 0.0)

        danger_answer = answers.get("danger", {})
        danger_prob = danger_answer.get("noul", 0.5)

        conf_answer = answers.get("confidence", {})
        conf_score = conf_answer.get("score", 2.0)

        script = script_library.get(situation_id)
        script_name = script.name if script else situation_id
        thought = self._generate_thought(state_text, situation_id, script_name, danger_prob, conf_score)

        if self.logger:
            self.logger.info(
                "Jev: situation=%s confidence=%.2f danger=%.2f latency=%.0fms",
                situation_id, confidence_val, danger_prob, elapsed_ms,
            )

        return {
            "situation": situation_id,
            "script_id": situation_id,
            "script_name": script_name,
            "confidence": confidence_val,
            "confidence_score": conf_score,
            "danger_prob": danger_prob,
            "probabilities": probabilities,
            "latency_ms": elapsed_ms,
            "thought": thought,
            "state_text": state_text,
        }

    def _generate_thought(
        self,
        state_text: str,
        situation_id: str,
        script_name: str,
        danger_prob: float,
        conf_score: float,
    ) -> str:
        """Generate a human-readable thought string for N.E.K.O personality layer."""
        parts = []

        if danger_prob > 0.7:
            parts.append("Dangerous! ")
        elif danger_prob > 0.4:
            parts.append("Risky... ")

        if conf_score >= 4:
            parts.append(f"Clearly {script_name}. ")
        elif conf_score >= 3:
            parts.append(f"Looks like {script_name}. ")
        elif conf_score >= 2:
            parts.append(f"Maybe {script_name}? ")
        else:
            parts.append(f"Not sure, trying {script_name}. ")

        if "spike" in state_text.lower():
            parts.append("Spikes nearby, need to be careful. ")
        if "has_dash" in state_text and "dash" in situation_id:
            parts.append("Dash is ready. ")
        if "climbing" in state_text.lower():
            parts.append("On the wall. ")
        if "airborne" in state_text.lower() or "Status: airborne" in state_text:
            parts.append("In the air. ")

        return "".join(parts).strip()

    def _fallback(
        self,
        state: CelesteState | dict | list,
        script_library: ScriptLibrary,
        error: str,
    ) -> dict[str, Any]:
        """Return a safe default when Jev is unavailable."""
        if isinstance(state, list):
            state = CelesteState.from_array(state)
        elif isinstance(state, dict):
            state = CelesteState(
                pos_x=state.get("pos_x", 0),
                pos_y=state.get("pos_y", 0),
                on_ground=state.get("on_ground", True),
                has_dash=state.get("has_dash", True),
            )

        if state.on_ground and state.has_dash:
            default = "simple_move_right"
        elif state.has_dash:
            default = "dash_across_gap"
        else:
            default = "rest_and_recover"

        return {
            "situation": default,
            "script_id": default,
            "script_name": default,
            "confidence": 0.0,
            "danger_prob": 0.5,
            "probabilities": {},
            "latency_ms": 0,
            "thought": f"(fallback) {default}",
            "state_text": self._encoder.encode(state),
            "error": error,
        }

    async def evaluate_action(
        self,
        state: CelesteState,
        action_name: str,
    ) -> dict[str, Any]:
        """Score an action for reward shaping (used by DQN).

        This is a compatibility method for the reward shaper.
        """
        state_text = self._encoder.encode(state)
        questions = {
            "is_dangerous": {
                "type": "noul",
                "instructions": (
                    f"Is performing '{action_name}' in this state immediately dangerous? "
                    "(will hit spikes, fall into pit, or waste dash)"
                ),
            },
            "action_quality": {
                "type": "score",
                "instructions": (
                    f"How reasonable is '{action_name}' for making progress "
                    "toward the level exit in this state?"
                ),
                "criteria": [
                    "will_die", "wrong_direction", "neutral", "good_progress", "optimal",
                ],
            },
        }

        await self._ensure_client()
        payload = {
            "model": self.model,
            "state": state_text,
            "questions": questions,
        }

        t0 = time.perf_counter()
        try:
            resp = await self._client.post(self.endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            if self.logger:
                self.logger.error("Jev evaluate_action failed: %s", e)
            return {"danger_prob": 0.5, "quality_score": 2.0, "shaped_reward": 0.0}

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._call_count += 1
        self._total_latency += elapsed_ms

        answers = data.get("answers", {})
        danger = answers.get("is_dangerous", {}).get("noul", 0.5)
        quality = answers.get("action_quality", {}).get("score", 2.0)

        penalty = -0.5 * max(0, danger - 0.5)
        bonus = 0.3 * max(0, quality - 2.0)
        shaped = penalty + bonus

        return {
            "danger_prob": danger,
            "quality_score": quality,
            "shaped_reward": shaped,
        }

    def get_stats(self) -> dict[str, Any]:
        avg = (self._total_latency / self._call_count * 1000) if self._call_count else 0
        return {
            "total_calls": self._call_count,
            "avg_latency_ms": avg,
            "est_cost_usd": self._call_count * 0.000194,
        }

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    def CONFENCE_QUESTION_ALT(self):
        return self.CONFIDENCE_QUESTION
