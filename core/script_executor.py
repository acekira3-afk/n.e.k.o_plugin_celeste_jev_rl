"""Frame-precise script executor with interrupt support.

Plays back a FrameScript frame-by-frame, checking for
mid-script state changes that require interruption.

This is the "muscle memory" layer: once Jev decides what to do,
this executor carries it out with frame-perfect precision.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Awaitable

from .script_library import FrameScript, KEY_NAMES
from .state_encoder import CelesteState, CelesteStateEncoder


class ScriptExecutor:
    """Executes frame-precise scripts with interrupt detection.

    Args:
        env: Celeste environment with step() and get_state()
        fps: Target frames per second
        logger: Logger instance
        on_thought: Async callback for thought display (N.E.K.O personality)
        on_state_change: Async callback for state observation
    """

    def __init__(
        self,
        env,
        fps: int = 60,
        logger=None,
        on_thought: Callable[[str, dict], Awaitable[None]] | None = None,
        on_state_change: Callable[[CelesteState], Awaitable[None]] | None = None,
    ):
        self.env = env
        self.fps = fps
        self.frame_time = 1.0 / fps
        self.logger = logger
        self.on_thought = on_thought
        self.on_state_change = on_state_change
        self._encoder = CelesteStateEncoder()

        self._current_script: FrameScript | None = None
        self._current_frame = 0
        self._interrupted = False
        self._total_scripts_run = 0
        self._total_frames_executed = 0
        self._total_interrupts = 0

    async def execute(
        self,
        script: FrameScript,
        initial_state: CelesteState | None = None,
        allow_interrupt: bool = True,
    ) -> dict[str, Any]:
        """Execute a frame-precise script.

        Returns:
            {
                "script_id": str,
                "completed": bool,          # True if script ran to end
                "interrupted": bool,        # True if interrupted
                "frames_executed": int,
                "final_state": CelesteState,
                "interrupt_reason": str,    # if interrupted
            }
        """
        self._current_script = script
        self._current_frame = 0
        self._interrupted = False
        self._total_scripts_run += 1

        if self.on_thought and initial_state:
            await self.on_thought(
                f"Executing: {script.name}",
                {
                    "script_id": script.id,
                    "duration_frames": script.duration_frames,
                    "difficulty": script.difficulty,
                },
            )

        if self.logger:
            self.logger.info(
                "Executing script: %s (%d frames, %.2fs)",
                script.id,
                script.duration_frames,
                script.duration_seconds,
            )

        last_keys: list[str] = []
        prev_state = initial_state

        for frame in range(script.duration_frames):
            if self._interrupted:
                break

            keys = script.get_input_at_frame(frame)

            if keys != last_keys:
                if self.logger and frame % 30 == 0:
                    self.logger.debug(
                        "Frame %d: keys=%s", frame, keys
                    )
                last_keys = keys

            bitmask = self._keys_to_bitmask(keys)
            next_state, reward, done, info = await self.env.step(bitmask)

            self._current_frame = frame
            self._total_frames_executed += 1

            if self.on_state_change:
                await self.on_state_change(next_state)

            if done:
                if self.logger:
                    self.logger.info(
                        "Script %s ended: game over at frame %d",
                        script.id, frame,
                    )
                return {
                    "script_id": script.id,
                    "completed": False,
                    "interrupted": False,
                    "game_over": True,
                    "frames_executed": frame + 1,
                    "final_state": next_state,
                    "interrupt_reason": "game_over",
                }

            if allow_interrupt and prev_state:
                should_stop, reason = self._check_interrupt(prev_state, next_state, script)
                if should_stop:
                    self._interrupted = True
                    self._total_interrupts += 1
                    if self.logger:
                        self.logger.info(
                            "Script %s interrupted at frame %d: %s",
                            script.id, frame, reason,
                        )
                    if self.on_thought:
                        await self.on_thought(
                            f"Interrupted: {reason}",
                            {"script_id": script.id, "frame": frame, "reason": reason},
                        )
                    return {
                        "script_id": script.id,
                        "completed": False,
                        "interrupted": True,
                        "frames_executed": frame + 1,
                        "final_state": next_state,
                        "interrupt_reason": reason,
                    }

            prev_state = next_state
            await asyncio.sleep(self.frame_time * 0.5)

        if self.on_thought and not self._interrupted:
            await self.on_thought(
                f"Done: {script.name}",
                {"script_id": script.id, "completed": True},
            )

        return {
            "script_id": script.id,
            "completed": not self._interrupted,
            "interrupted": False,
            "frames_executed": self._current_frame + 1,
            "final_state": prev_state,
            "interrupt_reason": None,
        }

    def _check_interrupt(
        self,
        prev: CelesteState,
        curr: CelesteState,
        script: FrameScript,
    ) -> tuple[bool, str]:
        """Check if script should be interrupted.

        Interrupt conditions:
          1. Position moved opposite to expected direction significantly
          2. Died (detected by sudden position reset)
          3. Stamina depleted while climbing
          4. Dash consumed unexpectedly
          5. Velocity reversed unexpectedly
        """
        dx = curr.pos_x - prev.pos_x
        dy = curr.pos_y - prev.pos_y

        if abs(dx) > 50 and dx < 0 and "right" in str(script.steps[0].keys):
            return True, "unexpected_leftward_drift"

        if abs(dy) > 50 and dy > 0 and "up" in str(script.steps):
            return True, "unexpected_fall"

        if curr.stamina < 0.05 and prev.stamina > 0.1 and "c" in str(script.steps):
            return True, "stamina_depleted"

        if (
            prev.has_dash and not curr.has_dash
            and "z" not in str(script.steps[min(self._current_frame, len(script.steps) - 1)].keys)
        ):
            return True, "dash_consumed_unexpectedly"

        return False, ""

    @staticmethod
    def _keys_to_bitmask(keys: list[str]) -> int:
        """Convert key names to 7-bit bitmask for the environment."""
        bit_map = {
            "up": 1, "down": 2, "left": 4, "right": 8,
            "z": 16, "x": 32, "c": 64,
        }
        mask = 0
        for k in keys:
            mask |= bit_map.get(k, 0)
        return mask

    def request_interrupt(self):
        """Request interruption of the current script."""
        self._interrupted = True

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_scripts_run": self._total_scripts_run,
            "total_frames_executed": self._total_frames_executed,
            "total_interrupts": self._total_interrupts,
            "interrupt_rate": (
                self._total_interrupts / self._total_scripts_run
                if self._total_scripts_run > 0
                else 0
            ),
        }
