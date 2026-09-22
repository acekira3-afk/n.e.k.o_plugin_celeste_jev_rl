"""Celeste game environment wrapper.

Supports Celeste Classic (PICO-8) via browser automation
and Celeste Full via mod API. Provides a gym-like interface.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from .state_encoder import CelesteState


class CelesteEnv:
    """Celeste game environment with async interface.

    Supports two backends:
      1. celeste_classic: PICO-8 web version via browser automation
      2. celeste_full: Celeste with Everest mod via Lua API
    """

    def __init__(
        self,
        target: str = "celeste_classic",
        rom_path: str = "",
        fps: int = 60,
        logger=None,
    ):
        self.target = target
        self.rom_path = rom_path
        self.fps = fps
        self.logger = logger
        self._running = False
        self._frame_time = 1.0 / fps
        self._episode_count = 0
        self._step_count = 0
        self._last_state: CelesteState | None = None
        self._browser = None

    async def reset(self) -> CelesteState:
        """Reset environment to start of level."""
        self._episode_count += 1
        self._step_count = 0
        if self.target == "celeste_classic":
            state = await self._reset_classic()
        else:
            state = await self._reset_full()
        self._last_state = state
        return state

    async def step(self, action: int) -> tuple[CelesteState, float, bool, dict]:
        """Take one step. Returns (next_state, reward, done, info)."""
        if self.target == "celeste_classic":
            result = await self._step_classic(action)
        else:
            result = await self._step_full(action)

        state, reward, done, info = result
        self._step_count += 1
        self._last_state = state
        return state, reward, done, info

    async def get_state(self) -> CelesteState:
        """Get current state without taking action."""
        if self._last_state:
            return self._last_state
        return await self.reset()

    async def close(self):
        self._running = False
        if self._browser:
            await self._browser.close()
            self._browser = None

    # ── Celeste Classic (PICO-8) ────────────────────────────────

    async def _reset_classic(self) -> CelesteState:
        """Reset PICO-8 Celeste Classic via browser automation.

        Sends keypress 'r' to restart the level, then reads
        the game state from the PICO-8 memory map.
        """
        self._running = True
        await self._press_keys(["r"], duration=0.05)
        await asyncio.sleep(0.5)
        return await self._read_pico8_state()

    async def _step_classic(self, action: int) -> tuple[CelesteState, float, bool, dict]:
        """Execute action on PICO-8 Celeste Classic.

        action: 0-127, bitmask of (up, down, left, right, z, x, c)
        """
        keys = self._action_to_keys(action)
        await self._press_keys(keys, duration=self._frame_time)

        state = await self._read_pico8_state()

        reward = self._compute_reward(state)
        done = self._check_done(state)
        info = {"step": self._step_count, "keys": keys}

        return state, reward, done, info

    async def _read_pico8_state(self) -> CelesteState:
        """Read game state from PICO-8 memory.

        In production, this reads from the PICO-8 JavaScript API
        or via a custom cartridge that exports state as JSON.
        For development, returns a placeholder state.
        """
        # TODO: Implement actual PICO-8 state reading via browser JS eval
        # For now, return a minimal state for development
        return CelesteState(
            pos_x=0.0, pos_y=0.0,
            vel_x=0.0, vel_y=0.0,
            obstacle_distances=[100.0] * 8,
            obstacle_types=[0] * 16,
            stamina=1.0,
            dist_to_goal=500.0,
            on_ground=True,
            has_dash=True,
        )

    # ── Celeste Full (Everest mod) ──────────────────────────────

    async def _reset_full(self) -> CelesteState:
        """Reset Celeste Full via Everest mod Lua API."""
        # Everest mod exposes state via a custom Lua module
        # that writes to a shared file or socket
        await self._send_celeste_command("reset")
        await asyncio.sleep(0.5)
        return await self._read_celeste_state()

    async def _step_full(self, action: int) -> tuple[CelesteState, float, bool, dict]:
        keys = self._action_to_keys(action)
        await self._send_celeste_command("input", {"keys": keys})
        await asyncio.sleep(self._frame_time)
        state = await self._read_celeste_state()
        reward = self._compute_reward(state)
        done = self._check_done(state)
        return state, reward, done, {"step": self._step_count, "keys": keys}

    async def _read_celeste_state(self) -> CelesteState:
        """Read state from Celeste Full via Everest mod."""
        # TODO: Implement actual Everest mod communication
        return CelesteState()

    async def _send_celeste_command(self, cmd: str, data: dict | None = None):
        """Send command to Celeste via Everest mod IPC."""
        # TODO: Implement via named pipe or socket
        pass

    # ── Utilities ───────────────────────────────────────────────

    def _action_to_keys(self, action: int) -> list[str]:
        """Convert action bitmask to key names.

        Bits: [up, down, left, right, z(dash), x(jump), c(grab)]
        """
        key_names = ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "z", "x", "c"]
        return [key_names[i] for i in range(7) if action & (1 << i)]

    async def _press_keys(self, keys: list[str], duration: float = 0.016):
        """Press keys via browser automation or system input."""
        # TODO: Integrate with N.E.K.O Browser Use channel or pyautogui
        if self._browser:
            for key in keys:
                await self._browser.press_key(key)
            if duration > 0:
                await asyncio.sleep(duration)
            for key in keys:
                await self._browser.release_key(key)

    def _compute_reward(self, state: CelesteState) -> float:
        """Reward function from Madeline's Policy Climb.

        fitness = num_levels_completed - distance_to_end_of_level

        Distance measured from last position to top-right corner.
        """
        if self._last_state is None:
            return 0.0
        prev_dist = self._last_state.dist_to_goal
        curr_dist = state.dist_to_goal
        progress = (prev_dist - curr_dist) / max(prev_dist, 1.0)
        return progress

    def _check_done(self, state: CelesteState) -> bool:
        """Check if episode is done."""
        if state.dist_to_goal < 5.0:
            return True
        if state.pos_y < -100:
            return True
        return False

    def get_stats(self) -> dict[str, Any]:
        return {
            "episodes": self._episode_count,
            "total_steps": self._step_count,
            "target": self.target,
            "fps": self.fps,
        }
