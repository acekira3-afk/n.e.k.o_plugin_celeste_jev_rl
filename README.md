# Celeste Jev RL — N.E.K.O Plugin

> Jev 情境识别 + 帧精确脚本执行 + N.E.K.O 人格表现，用于蔚蓝（Celeste）闯关实验

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![N.E.K.O Plugin](https://img.shields.io/badge/N.E.K.O-Plugin-purple.svg)](https://project-neko.online/)
[![Tests](https://img.shields.io/badge/tests-36%20passed-brightgreen.svg)]()

## 核心理念

将蔚蓝速通者的操作模式拆解为三阶段循环：

```
感知 → 思考 → 执行 → 感知 → ...
```

| 阶段 | 执行者 | 耗时 | 说明 |
|------|--------|------|------|
| 感知 | CelesteEnv | 1-3帧 | 读取游戏状态（位置、障碍物、体力等） |
| 思考 | Jev + N.E.K.O | 0.3-1秒 | Jev 识别情境选择脚本，N.E.K.O 显示表情和独白 |
| 执行 | ScriptExecutor | 脚本长度 | 帧精确回放操作序列，支持中途打断 |

**关键洞察**：Jev 的 70-500ms 延迟不再是缺陷，而是"角色思考时间"——就像真实玩家"看一眼→想一下→搓一套"。

## 架构

```
┌─ 感知 ─────────────────────────────────────┐
│  CelesteEnv → CelesteState (26维)          │
│  CelesteStateEncoder → 结构化文本           │
└──────────────┬────────────────────────────┘
               ▼
┌─ 思考（Jev + N.E.K.O 人格层）──────────────┐
│  JevSituationAgent                          │
│    → Choice: "这是什么情境？"               │
│    → Noul: "当前状态危险吗？"               │
│    → Score: "判断置信度？"                   │
│  PersonalityLayer                           │
│    → Live2D 表情（思考/专注/恐慌/开心）      │
│    → 独白文字（"右边有尖刺...得先跳"）       │
│    → 位置记忆（"上次在这死了，换条路"）      │
│  思考暂停 0.3-1.0秒（拟人反应时间）         │
└──────────────┬────────────────────────────┘
               ▼
┌─ 执行（帧精确脚本）────────────────────────┐
│  ScriptExecutor                            │
│    → 逐帧回放 FrameScript                  │
│    → 中途打断检测（位置漂移/死亡/体力耗尽） │
│    → 打断后回到感知阶段                     │
│  ScriptLibrary                             │
│    → 10 个内置脚本（冲刺间隙/蹬墙/波冲等）  │
│    → 支持自定义脚本录制                     │
│    → 使用统计（成功率追踪）                 │
└────────────────────────────────────────────┘
```

## 内置脚本库

| 脚本 ID | 名称 | 难度 | 说明 |
|---------|------|------|------|
| `simple_move_right` | 向右移动 | easy | 平地移动 |
| `dash_across_gap` | 冲刺过间隙 | normal | 跑→跳→空中冲刺右上 |
| `wall_jump_climb` | 蹬墙攀爬 | hard | 抓墙→爬→蹬墙跳→冲刺 |
| `spike_corridor_wait` | 尖刺走廊等待 | normal | 等移动尖刺通过后跑过 |
| `wave_dash_chain` | 波冲连段 | extreme | 连续波冲获得最大速度 |
| `ceiling_dash_traverse` | 天花板冲刺 | extreme | 沿天花板冲刺避地刺 |
| `strawberry_collect` | 收集草莓 | hard | 接近→收集→返回安全区 |
| `rest_and_recover` | 休息恢复 | easy | 安全点恢复体力 |
| `hyper_jump` | 超级跳 | hard | 下冲刺→跳获得爆发速度 |
| `backtrack_left` | 向左回退 | easy | 回退寻找替代路径 |

## 人格表现层

### 9 种 Live2D 表情

| 表情 | 触发条件 |
|------|---------|
| `neutral` | 初始/安全状态 |
| `thinking` | Jev 正在识别情境 |
| `focused` | 执行 normal 难度脚本 |
| `intense` | 执行 hard/extreme 脚本 |
| `panicked` | 危险度 > 0.7 |
| `happy` | 脚本成功完成 |
| `frustrated` | 脚本失败/中断 |
| `surprised` | 状态突变 |
| `tired` | 休息恢复中 |

### 独白系统

每个阶段生成符合情境的独白文字：

```
[思考] "让我看看... 前方有间隙，冲刺可用。"
       → 选择 dash_across_gap

[执行] "来了！" (表情: focused)

[成功] "好！" (表情: happy)

[失败] "不！上次在这选了 dash_across_gap 失败了，试试 wall_jump_climb"
       (位置记忆触发)
```

### 位置记忆

以 20px 网格量化位置，记录每个位置的使用历史和结果。当角色再次到达同一位置时，触发记忆独白。

## 项目结构

```
celeste-jev-rl/
├── plugin.toml                    # N.E.K.O 插件清单
├── config.example.toml            # 配置模板
├── __init__.py                    # 插件主入口（6 entries + 3 LLM tools）
├── core/
│   ├── celeste_env.py             # 蔚蓝游戏环境
│   ├── jev_agent.py               # Jev 情境识别 agent
│   ├── script_library.py          # 帧精确脚本库（10 内置脚本）
│   ├── script_executor.py         # 脚本执行器（中断支持）
│   ├── personality.py              # N.E.K.O 人格表现层
│   ├── state_encoder.py           # 状态编码器
│   ├── dqn_agent.py               # DQN 训练层（兼容保留）
│   ├── reward_shaper.py           # Jev 奖励塑形器（兼容保留）
│   └── experiment_runner.py       # Think-Execute 实验调度器
├── tests/
│   └── test_plugin.py             # 36 个单元测试
├── .github/workflows/
│   ├── verify.yml                 # CI: 测试 + 导入检查
│   └── release.yml                # CD: 标签触发发布
├── docs/
│   └── experiment_design.md       # 实验方案
├── conftest.py                    # 测试配置
├── .gitignore
├── LICENSE
└── README.md
```

## 快速开始

### 前置条件

1. **N.E.K.O** 桌面应用
2. **Jev API Key** — 从 [TypeSafe AI](https://typesafe.ai) 获取
3. Python 3.10+ 依赖：`numpy torch httpx pytest pytest-asyncio`

### 安装

1. 在 N.E.K.O 插件管理器中启用**开发者模式**
2. **加载未打包插件**，选择本目录
3. 复制 `config.example.toml` 为 `config.toml`，填入 Jev API key
4. 启动插件

### 运行

```python
# 在 N.E.K.O 对话中
> 运行蔚蓝实验，100 episodes

# 或通过 API
POST /api/plugins/celeste_jev_rl/run_experiment
{"episodes": 100}
```

### 查看角色状态

```python
# 获取当前表情和思维
> 蔚蓝角色现在什么状态？

# 返回
{
  "expression": "focused",
  "recent_thoughts": [
    {"text": "来了！", "expression": "focused"},
    {"text": "前方有间隙，冲刺可用。", "expression": "thinking"}
  ]
}
```

## LLM Tools

| 工具 | 说明 |
|------|------|
| `celeste_get_state` | 获取当前游戏状态 |
| `celeste_jev_decide` | 让 Jev 识别当前情境并选择脚本 |
| `celeste_get_thoughts` | 获取角色最近思维和当前表情 |

## 开发路线

- [x] 三段式架构（感知-思考-执行）
- [x] 帧精确脚本库（10 内置脚本）
- [x] 脚本执行器（中断检测）
- [x] N.E.K.O 人格表现层（9 表情 + 独白 + 位置记忆）
- [x] Jev 情境识别 agent
- [x] 36 个单元测试全部通过
- [ ] PICO-8 状态读取 cartridge
- [ ] Browser Use 集成
- [ ] 自定义脚本录制工具
- [ ] 实际运行实验
- [ ] DQN 脚本生成（用 DQN 学习新脚本并加入库）

## 许可证

MIT License

## 致谢

- [TypeSafe AI](https://typesafe.ai) — Jev 决策模型
- [Project N.E.K.O](https://github.com/Project-N-E-K-O/N.E.K.O) — 插件框架
- [sc2ad/CelesteBot](https://github.com/sc2ad/CelesteBot) — 蔚蓝 AI 先驱
- 蔚蓝速通社区 — 帧精确操作脚本参考
