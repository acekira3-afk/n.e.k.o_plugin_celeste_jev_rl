"""N.E.K.O personality layer for Celeste gameplay.

Transforms Jev's raw decisions into human-like expressions:
  - Live2D facial expressions (thinking, focused, panicked, happy)
  - Spoken thoughts and monologue
  - Memory feedback (remembering past failures at this position)
  - Emotional state tracking across episodes

This is what makes the agent feel like a "character" rather than
a bot: it pauses to think, shows emotion, and learns from mistakes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Expression(Enum):
    """Live2D expression states mapped to Celeste gameplay."""

    NEUTRAL = "neutral"
    THINKING = "thinking"        # Jev is deciding
    FOCUSED = "focused"          # Script executing, normal difficulty
    INTENSE = "intense"          # Hard/extreme script executing
    PANICKED = "panicked"        # Dangerous situation detected
    HAPPY = "happy"              # Script completed successfully
    FRUSTRATED = "frustrated"    # Script failed / interrupted
    SURPRISED = "surprised"      # Unexpected state change
    TIRED = "tired"              # Low stamina, resting


@dataclass
class ThoughtRecord:
    """A single thought moment for display."""

    timestamp: float
    text: str
    expression: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionMemory:
    """Memory of what happened at a specific game position."""

    pos_x: float
    pos_y: float
    script_id: str
    result: str  # "success" / "interrupted" / "game_over"
    timestamp: float
    thought: str


class PersonalityLayer:
    """Manages expressions, monologue, and memory for the Celeste agent.

    Integration points with N.E.K.O:
      - self.report_status() → N.E.K.O's self.report_status()
      - Live2D expression via N.E.K.O avatar system
      - Spoken thoughts via N.E.K.O voice synthesis (optional)
      - Memory via N.E.K.O five-dimensional memory system
    """

    # Expression mapping based on game state
    DANGER_EXPRESSION_MAP = [
        (0.8, Expression.PANICKED),
        (0.5, Expression.INTENSE),
        (0.3, Expression.FOCUSED),
        (0.0, Expression.NEUTRAL),
    ]

    DIFFICULTY_EXPRESSION_MAP = {
        "easy": Expression.NEUTRAL,
        "normal": Expression.FOCUSED,
        "hard": Expression.INTENSE,
        "extreme": Expression.INTENSE,
    }

    # Monologue templates
    THOUGHT_TEMPLATES = {
        "thinking": [
            "Let me see...",
            "What's ahead...",
            "Hold on, let me think...",
            "Okay, what do we have here...",
        ],
        "dangerous": [
            "That's close!",
            "Careful, careful...",
            "One wrong move and...",
            "This is risky...",
        ],
        "executing": [
            "Here goes!",
            "Let's do this!",
            "Focus...",
            "I've got this.",
        ],
        "success": [
            "Yes!",
            "That worked!",
            "Nice!",
            "On to the next.",
        ],
        "failure": [
            "No!",
            "That didn't work...",
            "So close!",
            "Again, again.",
        ],
        "rest": [
            "Phew...",
            "Let me catch my breath.",
            "Okay, what's next...",
            "Safe for now.",
        ],
        "remember": [
            "Wait, last time here I...",
            "I remember this spot...",
            "This didn't work before...",
            "Let me try something different.",
        ],
    }

    def __init__(self, logger=None):
        self.logger = logger
        self._expression = Expression.NEUTRAL
        self._thoughts: list[ThoughtRecord] = []
        self._position_memories: dict[str, PositionMemory] = {}
        self._episode_emotion_log: list[dict] = []
        self._death_count = 0
        self._success_count = 0
        self._current_episode = 0

    @property
    def expression(self) -> Expression:
        return self._expression

    def set_expression(self, expr: Expression):
        """Update Live2D expression."""
        if expr != self._expression:
            self._expression = expr
            if self.logger:
                self.logger.debug("Expression: %s", expr.value)

    async def on_thinking(
        self,
        state_text: str,
        jev_result: dict[str, Any],
    ) -> str:
        """Called when Jev is recognizing the situation.

        Shows thinking expression and generates monologue.
        Returns the thought string for display.
        """
        danger = jev_result.get("danger_prob", 0.5)
        conf = jev_result.get("confidence_score", 2.0)

        if danger > 0.7:
            self.set_expression(Expression.PANICKED)
            templates = self.THOUGHT_TEMPLATES["dangerous"]
        elif conf >= 4:
            self.set_expression(Expression.THINKING)
            templates = self.THOUGHT_TEMPLATES["thinking"]
        else:
            self.set_expression(Expression.THINKING)
            templates = self.THOUGHT_TEMPLATES["thinking"]

        import random
        base_thought = random.choice(templates)
        jev_thought = jev_result.get("thought", "")

        thought = f"{base_thought} {jev_thought}".strip()

        self._record_thought(thought, {
            "phase": "thinking",
            "danger_prob": danger,
            "confidence": conf,
            "situation": jev_result.get("situation", ""),
        })

        return thought

    async def on_executing(
        self,
        script_id: str,
        script_name: str,
        difficulty: str,
        danger_prob: float,
    ) -> str:
        """Called when a script starts executing."""
        if danger_prob > 0.5:
            self.set_expression(Expression.PANICKED)
        else:
            expr = self.DIFFICULTY_EXPRESSION_MAP.get(difficulty, Expression.FOCUSED)
            self.set_expression(expr)

        import random
        thought = random.choice(self.THOUGHT_TEMPLATES["executing"])

        self._record_thought(thought, {
            "phase": "executing",
            "script_id": script_id,
            "difficulty": difficulty,
        })

        return thought

    async def on_success(self, script_id: str, script_name: str) -> str:
        """Called when a script completes successfully."""
        self.set_expression(Expression.HAPPY)
        self._success_count += 1

        import random
        thought = random.choice(self.THOUGHT_TEMPLATES["success"])

        self._record_thought(thought, {
            "phase": "success",
            "script_id": script_id,
        })

        return thought

    async def on_failure(
        self,
        script_id: str,
        reason: str,
        position: tuple[float, float] | None = None,
    ) -> str:
        """Called when a script fails or is interrupted."""
        self.set_expression(Expression.FRUSTRATED)
        self._death_count += 1

        import random
        thought = random.choice(self.THOUGHT_TEMPLATES["failure"])

        if position:
            key = self._pos_key(position)
            if key in self._position_memories:
                prev = self._position_memories[key]
                remember = random.choice(self.THOUGHT_TEMPLATES["remember"])
                thought = f"{thought} {remember} (last time: {prev.script_id} → {prev.result})"

            self._position_memories[key] = PositionMemory(
                pos_x=position[0],
                pos_y=position[1],
                script_id=script_id,
                result=reason,
                timestamp=time.time(),
                thought=thought,
            )

        self._record_thought(thought, {
            "phase": "failure",
            "script_id": script_id,
            "reason": reason,
        })

        return thought

    async def on_rest(self) -> str:
        """Called during rest periods."""
        self.set_expression(Expression.TIRED)

        import random
        thought = random.choice(self.THOUGHT_TEMPLATES["rest"])

        self._record_thought(thought, {"phase": "rest"})
        return thought

    def check_position_memory(self, pos_x: float, pos_y: float) -> str | None:
        """Check if we've been in this position before.

        Returns a memory thought if we have history here.
        """
        key = self._pos_key((pos_x, pos_y))
        if key in self._position_memories:
            mem = self._position_memories[key]
            import random
            remember = random.choice(self.THOUGHT_TEMPLATES["remember"])
            return f"{remember} (last: {mem.script_id} → {mem.result})"
        return None

    def _record_thought(self, text: str, context: dict[str, Any]):
        record = ThoughtRecord(
            timestamp=time.time(),
            text=text,
            expression=self._expression.value,
            context=context,
        )
        self._thoughts.append(record)
        if len(self._thoughts) > 200:
            self._thoughts = self._thoughts[-100:]

        if self.logger:
            self.logger.info("[Thought] %s (expr: %s)", text, self._expression.value)

    @staticmethod
    def _pos_key(pos: tuple[float, float]) -> str:
        """Quantize position to grid for memory matching."""
        return f"{int(pos[0] // 20)},{int(pos[1] // 20)}"

    def start_episode(self, episode_num: int):
        self._current_episode = episode_num

    def end_episode(self, completed: bool):
        self._episode_emotion_log.append({
            "episode": self._current_episode,
            "completed": completed,
            "deaths": self._death_count,
            "successes": self._success_count,
            "expression_final": self._expression.value,
        })

    def get_recent_thoughts(self, n: int = 10) -> list[dict[str, Any]]:
        return [
            {
                "text": t.text,
                "expression": t.expression,
                "timestamp": t.timestamp,
                "context": t.context,
            }
            for t in self._thoughts[-n:]
        ]

    def get_personality_stats(self) -> dict[str, Any]:
        return {
            "current_expression": self._expression.value,
            "total_deaths": self._death_count,
            "total_successes": self._success_count,
            "position_memories": len(self._position_memories),
            "total_thoughts": len(self._thoughts),
            "episodes_logged": len(self._episode_emotion_log),
        }
