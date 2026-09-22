"""N.E.K.O Plugin: Celeste Jev RL

Three-phase Celeste gameplay: Perceive → Think → Execute.
Jev recognizes situations, selects frame-precise scripts,
N.E.K.O personality layer shows expressions and monologue.
"""

from typing import Any

try:
    from plugin.sdk.plugin import (
        NekoPluginBase,
        neko_plugin,
        plugin_entry,
        lifecycle,
        timer_interval,
        llm_tool,
        Ok,
        Err,
        SdkError,
    )
    _HAS_NEKO_SDK = True
except ImportError:
    _HAS_NEKO_SDK = False
    NekoPluginBase = object
    def neko_plugin(cls): return cls
    def plugin_entry(**kw):
        def deco(fn): return fn
        return deco
    def lifecycle(**kw):
        def deco(fn): return fn
        return deco
    def timer_interval(**kw):
        def deco(fn): return fn
        return deco
    def llm_tool(**kw):
        def deco(fn): return fn
        return deco
    Ok = lambda v: {"ok": v}
    Err = lambda v: {"err": str(v)}
    SdkError = Exception


@neko_plugin
class CelesteJevRLPlugin(NekoPluginBase):
    """Celeste Think-Execute plugin for N.E.K.O.

    Architecture:
      1. Perceive: Read game state (position, obstacles, mechanics)
      2. Think:    Jev recognizes situation → selects script
                   N.E.K.O shows expression + monologue
      3. Execute: Frame-precise script playback with interrupts
    """

    def __init__(self, ctx: Any = None):
        if _HAS_NEKO_SDK and ctx is not None:
            super().__init__(ctx)
        self.logger = getattr(ctx, "logger", None) if ctx else None
        self._runner = None
        self._env = None
        self._jev = None
        self._scripts = None
        self._personality = None
        self._dqn = None
        self._shaper = None

    def _init_components(self):
        from .core.celeste_env import CelesteEnv
        from .core.jev_agent import JevSituationAgent
        from .core.script_library import ScriptLibrary
        from .core.script_executor import ScriptExecutor
        from .core.personality import PersonalityLayer
        from .core.dqn_agent import DQNAgent
        from .core.reward_shaper import JevRewardShaper
        from .core.experiment_runner import ThinkExecuteRunner

        cfg = self.config if hasattr(self, "config") else {}

        self._scripts = ScriptLibrary(logger=self.logger)

        jev_cfg = cfg.get("jev", {}) if isinstance(cfg, dict) else {}
        if jev_cfg.get("api_key"):
            self._jev = JevSituationAgent(
                api_key=jev_cfg["api_key"],
                endpoint=jev_cfg.get("endpoint", "https://api.typesafe.ai/v1/systemone"),
                model=jev_cfg.get("model", "jev-latest"),
                logger=self.logger,
            )

        dqn_cfg = cfg.get("dqn", {}) if isinstance(cfg, dict) else {}
        self._dqn = DQNAgent(
            state_dim=26, action_dim=128,
            lr=dqn_cfg.get("learning_rate", 1e-4),
            gamma=dqn_cfg.get("gamma", 0.99),
            epsilon_start=dqn_cfg.get("epsilon_start", 1.0),
            epsilon_end=dqn_cfg.get("epsilon_end", 0.01),
            epsilon_decay=dqn_cfg.get("epsilon_decay", 0.995),
            buffer_size=dqn_cfg.get("buffer_size", 10000),
            batch_size=dqn_cfg.get("batch_size", 32),
            target_update=dqn_cfg.get("target_update", 10),
            logger=self.logger,
        )

        if self._jev:
            self._shaper = JevRewardShaper(jev=self._jev, logger=self.logger)

        self._personality = PersonalityLayer(logger=self.logger)

        game_cfg = cfg.get("game", {}) if isinstance(cfg, dict) else {}
        self._env = CelesteEnv(
            target=game_cfg.get("target", "celeste_classic"),
            rom_path=game_cfg.get("rom_path", ""),
            fps=game_cfg.get("fps", 60),
            logger=self.logger,
        )

        self._runner = ThinkExecuteRunner(
            env=self._env,
            jev=self._jev,
            script_library=self._scripts,
            dqn=self._dqn,
            shaper=self._shaper,
            personality=self._personality,
            logger=self.logger,
        )

    # ── Lifecycle ──────────────────────────────────────────────

    @lifecycle(id="startup")
    async def on_startup(self, **_):
        self._init_components()
        if self.logger:
            self.logger.info("CelesteJevRL ready (Think-Execute mode)")
        return Ok({"status": "ready"})

    @lifecycle(id="shutdown")
    async def on_shutdown(self, **_):
        if self._env:
            await self._env.close()
        if self._jev:
            await self._jev.close()
        if self.logger:
            self.logger.info("CelesteJevRL stopped")
        return Ok({"status": "stopped"})

    @lifecycle(id="reload")
    async def on_reload(self, **_):
        return Ok({"status": "reloaded"})

    # ── Plugin Entries ─────────────────────────────────────────

    @plugin_entry(
        id="run_experiment",
        name="Run Experiment",
        description="Run the Think-Execute loop for Celeste gameplay",
        input_schema={
            "type": "object",
            "properties": {
                "episodes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10000,
                    "description": "Number of episodes to run",
                },
            },
        },
    )
    async def run_experiment(self, episodes: int = 100, **_):
        if not self._runner:
            self._init_components()
        if not self._runner:
            return Err(SdkError("Plugin not initialized"))
        result = await self._runner.run(episodes=episodes)
        return Ok(result)

    @plugin_entry(
        id="get_status",
        name="Get Status",
        description="Get current experiment status, expression, and recent thoughts",
    )
    async def get_status(self, **_):
        if not self._runner:
            return Ok({"status": "idle"})
        return Ok(self._runner.get_status())

    @plugin_entry(
        id="stop_experiment",
        name="Stop Experiment",
        description="Stop the currently running experiment",
    )
    async def stop_experiment(self, **_):
        if not self._runner:
            return Err(SdkError("No experiment running"))
        await self._runner.stop()
        return Ok({"stopped": True})

    @plugin_entry(
        id="get_results",
        name="Get Results",
        description="Retrieve experiment results and personality thoughts",
        input_schema={
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["summary", "full"],
                    "default": "summary",
                },
            },
        },
    )
    async def get_results(self, format: str = "summary", **_):
        if not self._runner:
            return Err(SdkError("No results available"))
        return Ok(self._runner.get_results(fmt=format))

    @plugin_entry(
        id="list_scripts",
        name="List Scripts",
        description="List all available frame-precise scripts",
    )
    async def list_scripts(self, **_):
        if not self._scripts:
            self._init_components()
        if not self._scripts:
            return Err(SdkError("Not initialized"))
        scripts = [
            s.to_dict() for s in [self._scripts.get(sid) for sid in self._scripts.list_scripts()] if s
        ]
        return Ok({"scripts": scripts})

    @plugin_entry(
        id="get_personality",
        name="Get Personality",
        description="Get personality stats: expression, deaths, memories, thoughts",
    )
    async def get_personality(self, **_):
        if not self._personality:
            return Ok({"status": "no personality layer"})
        return Ok(self._personality.get_personality_stats())

    # ── LLM Tools ──────────────────────────────────────────────

    @llm_tool(
        name="celeste_get_state",
        description="Get the current Celeste game state as structured text.",
        parameters={"type": "object", "properties": {}},
    )
    async def celeste_get_state(self, **_):
        if not self._env:
            self._init_components()
        if not self._env:
            return {"error": "Game environment not initialized"}
        from .core.state_encoder import CelesteStateEncoder
        state = await self._env.get_state()
        encoder = CelesteStateEncoder()
        return {"state": encoder.encode(state), "raw": state.to_array()}

    @llm_tool(
        name="celeste_jev_decide",
        description="Ask Jev to recognize the current situation and select a script.",
        parameters={
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "string",
                    "description": "Not used in Think-Execute mode, kept for compatibility",
                },
            },
        },
    )
    async def celeste_jev_decide(self, *, strategy: str = "situation", **_):
        if not self._jev or not self._env or not self._scripts:
            self._init_components()
        if not self._jev or not self._env or not self._scripts:
            return {"error": "Jev or environment not available"}
        state = await self._env.get_state()
        result = await self._jev.recognize_situation(state, self._scripts)
        return result

    @llm_tool(
        name="celeste_get_thoughts",
        description="Get the agent's recent thoughts and current expression.",
        parameters={"type": "object", "properties": {}},
    )
    async def celeste_get_thoughts(self, **_):
        if not self._personality:
            return {"error": "Personality layer not initialized"}
        return {
            "expression": self._personality.expression.value,
            "recent_thoughts": self._personality.get_recent_thoughts(10),
            "stats": self._personality.get_personality_stats(),
        }
