"""Celeste game state encoder.

Converts raw game state (26 numeric inputs) into structured text
suitable for Jev's state input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

OBSTACLE_TYPES = {
    0: "empty",
    1: "spike",
    2: "wall",
    3: "platform",
    4: "spike_up",
    5: "spike_down",
    6: "spike_left",
    7: "spike_right",
    8: "crystal",
    9: "spring",
    10: "feather",
    11: " strawberry",
    12: "dash_refill",
    13: "exit_block",
    14: "moving_block",
    15: "trigger",
}

DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

ACTION_NAMES = {
    0: "none",
    1: "move_left",
    2: "move_right",
    3: "jump",
    4: "dash_up",
    5: "dash_right",
    6: "dash_up_right",
    7: "climb_up",
    8: "wall_jump",
    9: "grab_wall",
    10: "release_wall",
    11: "dash_down",
    12: "dash_left",
    13: "dash_down_right",
    14: "dash_up_left",
    15: "dash_down_left",
}

# Reduced action space: 16 most useful action combos
ACTION_SPACE = [
    "none",
    "move_left",
    "move_right",
    "jump",
    "move_left+jump",
    "move_right+jump",
    "dash_right",
    "dash_up_right",
    "dash_up",
    "grab_wall",
    "climb_up",
    "wall_jump",
    "move_left+grab",
    "move_right+grab",
    "dash_up_left",
    "wait",
]


@dataclass
class CelesteState:
    """Celeste game state, matching the 26-input specification."""

    pos_x: float = 0.0
    pos_y: float = 0.0
    vel_x: float = 0.0
    vel_y: float = 0.0
    obstacle_distances: list[float] = field(default_factory=lambda: [0.0] * 8)
    obstacle_types: list[int] = field(default_factory=lambda: [0] * 16)
    stamina: float = 1.0
    dist_to_goal: float = 0.0
    on_ground: bool = False
    swimming: bool = False
    climbing: bool = False
    has_dash: bool = True

    @classmethod
    def from_array(cls, arr: list[float]) -> CelesteState:
        s = cls()
        s.pos_x = arr[0]
        s.pos_y = arr[1]
        s.vel_x = arr[2]
        s.vel_y = arr[3]
        s.obstacle_distances = arr[4:12]
        s.obstacle_types = [int(x) for x in arr[12:28]] if len(arr) >= 28 else [0] * 16
        s.stamina = arr[28] if len(arr) > 28 else 1.0
        s.dist_to_goal = arr[29] if len(arr) > 29 else 0.0
        s.on_ground = bool(arr[30]) if len(arr) > 30 else False
        s.swimming = bool(arr[31]) if len(arr) > 31 else False
        s.climbing = bool(arr[32]) if len(arr) > 32 else False
        s.has_dash = bool(arr[33]) if len(arr) > 33 else True
        return s

    def to_array(self) -> list[float]:
        return [
            self.pos_x, self.pos_y,
            self.vel_x, self.vel_y,
            *self.obstacle_distances,
            *[float(t) for t in self.obstacle_types[:16]],
            self.stamina, self.dist_to_goal,
            float(self.on_ground), float(self.swimming),
            float(self.climbing), float(self.has_dash),
        ]


class CelesteStateEncoder:
    """Encodes CelesteState into text for Jev consumption."""

    def encode(self, state: CelesteState | dict | list) -> str:
        if isinstance(state, dict):
            state = CelesteState(
                pos_x=state.get("pos_x", 0),
                pos_y=state.get("pos_y", 0),
                vel_x=state.get("vel_x", 0),
                vel_y=state.get("vel_y", 0),
                obstacle_distances=state.get("obstacle_distances", [0] * 8),
                obstacle_types=state.get("obstacle_types", [0] * 16),
                stamina=state.get("stamina", 1.0),
                dist_to_goal=state.get("dist_to_goal", 0.0),
                on_ground=state.get("on_ground", False),
                swimming=state.get("swimming", False),
                climbing=state.get("climbing", False),
                has_dash=state.get("has_dash", True),
            )
        elif isinstance(state, list):
            state = CelesteState.from_array(state)
        elif not isinstance(state, CelesteState):
            raise TypeError(f"Unsupported state type: {type(state)}")

        parts: list[str] = []
        parts.append(f"Player position: ({state.pos_x:.1f}, {state.pos_y:.1f})")
        parts.append(f"Velocity: ({state.vel_x:.1f}, {state.vel_y:.1f})")

        obstacle_descs = []
        for i, (dist, otype) in enumerate(
            zip(state.obstacle_distances, state.obstacle_types[:8])
        ):
            if otype > 0:
                name = OBSTACLE_TYPES.get(otype, f"type_{otype}")
                obstacle_descs.append(f"{DIRECTIONS[i]}={dist:.0f}px({name})")
        if obstacle_descs:
            parts.append(f"Nearby obstacles: {' '.join(obstacle_descs)}")
        else:
            parts.append("Nearby obstacles: none within range")

        parts.append(f"Stamina: {state.stamina:.2f}")
        parts.append(f"Distance to goal: {state.dist_to_goal:.0f}")

        flags = []
        if state.on_ground:
            flags.append("on_ground")
        if state.climbing:
            flags.append("climbing")
        if state.has_dash:
            flags.append("dash_ready")
        if state.swimming:
            flags.append("swimming")
        parts.append(f"Status: {', '.join(flags) if flags else 'airborne'}")

        return " | ".join(parts)

    def encode_goal_context(self, state: CelesteState, goal: str) -> str:
        base = self.encode(state)
        return f"{base} | Current goal: {goal}"

    def get_action_options(self) -> dict[str, str]:
        return {str(i): name for i, name in enumerate(ACTION_SPACE)}

    def get_goal_options(self, state: CelesteState) -> dict[str, str]:
        goals = ["reach_exit"]
        if state.has_dash:
            goals.append("dash_across_gap")
        if state.climbing:
            goals.append("climb_up_wall")
        if state.on_ground and abs(state.vel_x) < 0.5:
            goals.append("jump_over_obstacle")
        if state.dist_to_goal > 200:
            goals.append("traverse_room")
        else:
            goals.append("reach_exit")
        if state.stamina < 0.3:
            goals.append("rest_on_ground")
        return {str(i): g for i, g in enumerate(goals)}

    def get_route_options(self, state: CelesteState) -> dict[str, str]:
        routes = [
            "path_a_direct",
            "path_b_upper",
            "path_c_lower",
            "backtrack",
        ]
        return {str(i): r for i, r in enumerate(routes)}
