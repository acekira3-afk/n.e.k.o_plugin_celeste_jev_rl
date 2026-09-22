"""Tests for Celeste Jev RL plugin — Think-Execute architecture."""

import asyncio
import pytest
import numpy as np
import torch

from core.state_encoder import CelesteState, CelesteStateEncoder, ACTION_SPACE
from core.dqn_agent import DQNAgent, ReplayBuffer, CelesteDQN
from core.reward_shaper import JevRewardShaper
from core.script_library import ScriptLibrary, FrameScript, FrameStep
from core.script_executor import ScriptExecutor
from core.personality import PersonalityLayer, Expression


class TestStateEncoder:
    def test_encode_basic(self):
        state = CelesteState(
            pos_x=100.5, pos_y=200.0,
            vel_x=3.0, vel_y=-1.5,
            obstacle_distances=[15.0, 12.0, 8.0, 20.0, 0.0, 0.0, 10.0, 14.0],
            obstacle_types=[2, 1, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            stamina=0.85,
            dist_to_goal=240.5,
            on_ground=False,
            has_dash=True,
        )
        encoder = CelesteStateEncoder()
        text = encoder.encode(state)
        assert "100.5" in text
        assert "200.0" in text
        assert "dash_ready" in text
        assert "240" in text

    def test_from_array(self):
        arr = [10.0, 20.0, 1.0, -1.0] + [15.0] * 8 + [0.0] * 16 + [1.0, 300.0, 1.0, 0.0, 0.0, 1.0]
        state = CelesteState.from_array(arr)
        assert state.pos_x == 10.0
        assert state.pos_y == 20.0
        assert state.on_ground is True
        assert state.has_dash is True

    def test_action_options(self):
        encoder = CelesteStateEncoder()
        options = encoder.get_action_options()
        assert len(options) >= 8
        assert "none" in options.values()

    def test_goal_options_vary_with_state(self):
        encoder = CelesteStateEncoder()
        state_climbing = CelesteState(climbing=True, stamina=0.2)
        state_grounded = CelesteState(on_ground=True, vel_x=0.0)
        goals_c = encoder.get_goal_options(state_climbing)
        goals_g = encoder.get_goal_options(state_grounded)
        assert goals_c != goals_g


class TestScriptLibrary:
    def test_builtin_scripts_loaded(self):
        lib = ScriptLibrary()
        assert "dash_across_gap" in lib.list_scripts()
        assert "wall_jump_climb" in lib.list_scripts()
        assert "wave_dash_chain" in lib.list_scripts()
        assert len(lib.list_scripts()) >= 10

    def test_get_script(self):
        lib = ScriptLibrary()
        script = lib.get("dash_across_gap")
        assert script is not None
        assert script.id == "dash_across_gap"
        assert len(script.steps) > 0
        assert script.duration_frames > 0

    def test_get_nonexistent_script(self):
        lib = ScriptLibrary()
        assert lib.get("nonexistent") is None

    def test_script_input_at_frame(self):
        script = FrameScript(
            id="test",
            name="Test",
            description="Test script",
            steps=[
                FrameStep(0, ["right"]),
                FrameStep(4, ["right", "x"]),
                FrameStep(6, ["right"]),
            ],
        )
        assert script.get_input_at_frame(0) == ["right"]
        assert script.get_input_at_frame(4) == ["right", "x"]
        assert script.get_input_at_frame(5) == ["right", "x"]
        assert script.get_input_at_frame(6) == ["right"]
        assert script.get_input_at_frame(100) == ["right"]

    def test_script_duration(self):
        script = FrameScript(
            id="test",
            name="Test",
            description="Test",
            steps=[
                FrameStep(0, ["right"]),
                FrameStep(10, []),
            ],
        )
        assert script.duration_frames == 11
        assert script.duration_seconds > 0

    def test_options_for_jev(self):
        lib = ScriptLibrary()
        options = lib.get_options_for_jev()
        assert len(options) >= 10
        for sid, desc in options.items():
            assert isinstance(sid, str)
            assert isinstance(desc, str)

    def test_record_usage(self):
        lib = ScriptLibrary()
        lib.record_usage("dash_across_gap", True)
        lib.record_usage("dash_across_gap", False)
        lib.record_usage("dash_across_gap", True)
        stats = lib.get_stats()
        assert stats["dash_across_gap"]["uses"] == 3
        assert stats["dash_across_gap"]["successes"] == 2
        assert abs(stats["dash_across_gap"]["success_rate"] - 2/3) < 0.01

    def test_export_json(self):
        lib = ScriptLibrary()
        json_str = lib.export_json()
        assert "dash_across_gap" in json_str
        assert "wave_dash_chain" in json_str

    def test_add_custom_script(self):
        lib = ScriptLibrary()
        custom = FrameScript(
            id="custom_test",
            name="Custom",
            description="Custom script",
            steps=[FrameStep(0, ["right"])],
        )
        lib.add_script(custom)
        assert "custom_test" in lib.list_scripts()
        assert lib.get("custom_test") is not None

    def test_prerequisites(self):
        lib = ScriptLibrary()
        script = lib.get("dash_across_gap")
        assert "has_dash" in script.prerequisites


class TestScriptExecutor:
    @pytest.mark.asyncio
    async def test_execute_simple_script(self):
        class MockEnv:
            fps = 60
            async def step(self, action):
                return CelesteState(pos_x=1.0, dist_to_goal=10.0), 0.1, False, {}
            async def get_state(self):
                return CelesteState()
            async def close(self):
                pass

        executor = ScriptExecutor(env=MockEnv(), fps=60)
        script = FrameScript(
            id="test",
            name="Test",
            description="Test",
            steps=[
                FrameStep(0, ["right"]),
                FrameStep(3, []),
            ],
        )
        result = await executor.execute(script, initial_state=CelesteState())
        assert result["script_id"] == "test"
        assert result["completed"] is True
        assert result["frames_executed"] > 0

    @pytest.mark.asyncio
    async def test_interrupt_on_game_over(self):
        class MockEnv:
            fps = 60
            async def step(self, action):
                return CelesteState(dist_to_goal=0.0), 1.0, True, {}
            async def get_state(self):
                return CelesteState()
            async def close(self):
                pass

        executor = ScriptExecutor(env=MockEnv(), fps=60)
        script = FrameScript(
            id="test_die",
            name="Test Die",
            description="Dies immediately",
            steps=[FrameStep(0, ["right"]), FrameStep(10, [])],
        )
        result = await executor.execute(script)
        assert result["game_over"] is True
        assert result["completed"] is False

    @pytest.mark.asyncio
    async def test_keys_to_bitmask(self):
        assert ScriptExecutor._keys_to_bitmask([]) == 0
        assert ScriptExecutor._keys_to_bitmask(["right"]) == 8
        assert ScriptExecutor._keys_to_bitmask(["right", "x"]) == 8 | 32
        assert ScriptExecutor._keys_to_bitmask(["z", "x", "c"]) == 16 | 32 | 64

    @pytest.mark.asyncio
    async def test_interrupt_request(self):
        class MockEnv:
            fps = 60
            def __init__(self):
                self._call_count = 0
            async def step(self, action):
                self._call_count += 1
                # After 3 frames, produce a big position drift to trigger interrupt
                if self._call_count > 3:
                    return CelesteState(pos_x=-100, dist_to_goal=10.0), 0.0, False, {}
                return CelesteState(pos_x=1.0, dist_to_goal=10.0), 0.0, False, {}
            async def get_state(self):
                return CelesteState()
            async def close(self):
                pass

        env = MockEnv()
        executor = ScriptExecutor(env=env, fps=60)
        script = FrameScript(
            id="test_long",
            name="Long",
            description="Long script",
            steps=[FrameStep(0, ["right"]), FrameStep(50, [])],
            duration_frames=50,
        )
        result = await executor.execute(script, initial_state=CelesteState(pos_x=0, dist_to_goal=10.0))
        # Should be interrupted due to position drift or at least not complete all 50 frames
        assert result["frames_executed"] < 50 or result["interrupted"] is True


class TestPersonality:
    @pytest.mark.asyncio
    async def test_thinking_expression(self):
        p = PersonalityLayer()
        await p.on_thinking("state text", {"danger_prob": 0.1, "confidence_score": 4, "thought": "test"})
        assert p.expression == Expression.THINKING

    @pytest.mark.asyncio
    async def test_panic_expression(self):
        p = PersonalityLayer()
        await p.on_thinking("state", {"danger_prob": 0.9, "confidence_score": 1, "thought": "danger!"})
        assert p.expression == Expression.PANICKED

    @pytest.mark.asyncio
    async def test_success_expression(self):
        p = PersonalityLayer()
        await p.on_success("dash_across_gap", "Dash Across Gap")
        assert p.expression == Expression.HAPPY

    @pytest.mark.asyncio
    async def test_failure_expression(self):
        p = PersonalityLayer()
        await p.on_failure("dash_across_gap", "game_over", position=(100, 200))
        assert p.expression == Expression.FRUSTRATED

    @pytest.mark.asyncio
    async def test_rest_expression(self):
        p = PersonalityLayer()
        await p.on_rest()
        assert p.expression == Expression.TIRED

    @pytest.mark.asyncio
    async def test_position_memory(self):
        p = PersonalityLayer()
        await p.on_failure("dash_across_gap", "game_over", position=(100, 200))
        memory = p.check_position_memory(100, 200)
        assert memory is not None
        assert "dash_across_gap" in memory

    @pytest.mark.asyncio
    async def test_no_memory_at_new_position(self):
        p = PersonalityLayer()
        assert p.check_position_memory(999, 999) is None

    def test_personality_stats(self):
        p = PersonalityLayer()
        stats = p.get_personality_stats()
        assert "current_expression" in stats
        assert "total_deaths" in stats
        assert "position_memories" in stats

    def test_recent_thoughts(self):
        p = PersonalityLayer()
        thoughts = p.get_recent_thoughts(5)
        assert isinstance(thoughts, list)
        assert len(thoughts) == 0

    @pytest.mark.asyncio
    async def test_thoughts_accumulate(self):
        p = PersonalityLayer()
        await p.on_thinking("state", {"danger_prob": 0.1, "confidence_score": 4, "thought": "test"})
        await p.on_success("s1", "Script 1")
        await p.on_failure("s2", "interrupted", position=(50, 50))
        thoughts = p.get_recent_thoughts(10)
        assert len(thoughts) >= 3


class TestDQN:
    def test_dqn_forward(self):
        net = CelesteDQN(state_dim=26, action_dim=128)
        x = torch.randn(1, 26)
        out = net(x)
        assert out.shape == (1, 128)

    def test_replay_buffer(self):
        buf = ReplayBuffer(100)
        for _ in range(50):
            buf.push(np.zeros(26), 0, 1.0, np.zeros(26), False)
        assert len(buf) == 50
        s, a, r, ns, d = buf.sample(16)
        assert len(s) == 16

    def test_dqn_agent_select_action(self):
        agent = DQNAgent(state_dim=26, action_dim=128, epsilon_start=1.0)
        state = np.random.randn(26).astype(np.float32)
        action = agent.select_action(state)
        assert 0 <= action < 128

    def test_dqn_agent_train(self):
        agent = DQNAgent(state_dim=26, action_dim=128, batch_size=4)
        for _ in range(10):
            agent.store_transition(
                np.random.randn(26), 0, 1.0, np.random.randn(26), False
            )
        loss = agent.train_step_update()
        assert loss is not None
        assert loss > 0

    def test_epsilon_decay(self):
        agent = DQNAgent(epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995)
        for _ in range(200):
            agent.end_episode()
        assert agent.epsilon < 0.5

    def test_save_load(self, tmp_path):
        agent = DQNAgent()
        agent.episode_count = 42
        path = str(tmp_path / "test_model.pt")
        agent.save(path)
        agent2 = DQNAgent()
        agent2.load(path)
        assert agent2.episode_count == 42


class TestRewardShaper:
    @pytest.mark.asyncio
    async def test_shaped_reward(self):
        class MockJev:
            async def evaluate_action(self, state, action_name):
                return {
                    "danger_prob": 0.9,
                    "quality_score": 1.0,
                    "shaped_reward": -0.4,
                }
            def get_stats(self):
                return {"total_calls": 0, "avg_latency_ms": 0, "est_cost_usd": 0}

        shaper = JevRewardShaper(jev=MockJev())
        state = CelesteState()
        reward = await shaper.shape(state, 0, "move_right", 1.0)
        assert reward < 1.0

    @pytest.mark.asyncio
    async def test_safe_action_bonus(self):
        class MockJev:
            async def evaluate_action(self, state, action_name):
                return {
                    "danger_prob": 0.1,
                    "quality_score": 4.5,
                    "shaped_reward": 0.15,
                }
            def get_stats(self):
                return {"total_calls": 0, "avg_latency_ms": 0, "est_cost_usd": 0}

        shaper = JevRewardShaper(jev=MockJev())
        state = CelesteState()
        reward = await shaper.shape(state, 0, "move_right", 1.0)
        assert reward > 1.0
