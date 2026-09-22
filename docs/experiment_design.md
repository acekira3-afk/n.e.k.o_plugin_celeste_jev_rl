# Celeste + Jev + DQN 实验方案

## 1. 研究问题

**核心问题**：Jev 决策模型能否加速或改进 DQN 在蔚蓝（Celeste）闯关中的强化学习训练？

**子问题**：
- Q1：零训练的 Jev 能否直接玩蔚蓝？能走多远？
- Q2：Jev 的语义判断能否作为 reward shaping 信号，加速 DQN 收敛？
- Q3：Jev 做高层路由 + DQN 做底层执行，是否优于纯 DQN？
- Q4：四种模式（Jev-only / DQN-only / Jev-shaped-DQN / Jev-router-DQN）的定量对比如何？

## 2. 四种实验模式

### 模式 A: Jev Only（纯 Jev 决策）

```
游戏状态 → StateEncoder → Jev API → 动作选择 → 游戏执行
```

- **零训练数据**，依赖 Jev 的零样本语义理解
- 每步调用 Jev API，延迟 70-500ms
- 预期能力：理解障碍物布局，但精确操作能力弱
- **验证点**：Jev 能否判断"该跳还是该冲刺"

### 模式 B: DQN Only（纯 DQN 基线）

```
游戏状态 → DQN 策略网络 → 动作选择 → 游戏执行
                         ↕
                    Replay Buffer 训练
```

- 标准 DQN，4层全连接网络（256→256→128→128）
- 26维输入，128维输出（7键的2^7组合）
- 奖励函数：`fitness = num_levels - distance_to_end`
- **基线**：参考论文 "Madeline's Policy Climb"，预期 1000 episode 通关第一关

### 模式 C: Jev Shaped DQN（Jev 奖励塑形 + DQN）

```
游戏状态 → DQN 选动作 → 游戏执行 → base_reward
                                    ↓
                    Jev 评估动作质量 → shaped_reward
                                    ↓
                            存入 Replay Buffer
```

- DQN 负责选动作和学策略
- Jev 负责评估每个 (state, action) 的危险性（noul）和合理性（score）
- 塑形奖励 = base_reward + penalty(danger) + bonus(quality)
- **核心假设**：Jev 的语义理解能帮助 DQN 避免明显错误探索，加速收敛

### 模式 D: Jev Router DQN（Jev 路由 + DQN 执行）

```
游戏状态 → Jev 选择战略目标（每10秒）
                ↓
        Jev 选择战术子目标（每5秒）
                ↓
        DQN 在当前目标下执行精确操作（每帧）
```

- 对应卡尼曼系统一/二理论：Jev 做直觉判断，DQN 做精确执行
- DQN 在不同目标下可能有不同策略网络
- **验证点**：高层语义路由是否减少 DQN 的探索空间

## 3. 实验参数

### 游戏环境
| 参数 | 值 | 说明 |
|------|-----|------|
| target | celeste_classic | PICO-8 版蔚蓝，状态空间更小 |
| fps | 60 | 目标帧率 |
| max_steps | 2000 | 每集最大步数 |

### DQN 超参数
| 参数 | 值 | 说明 |
|------|-----|------|
| state_dim | 26 | 论文标准输入维度 |
| action_dim | 128 | 7键2^7组合 |
| learning_rate | 1e-4 | Adam 优化器 |
| gamma | 0.99 | 折扣因子 |
| epsilon | 1.0→0.01 | 探索率衰减 |
| epsilon_decay | 0.995 | 每集衰减 |
| buffer_size | 10000 | 经验回放 |
| batch_size | 32 | 训练批次 |
| target_update | 10 | 目标网络更新频率 |

### Jev 配置
| 参数 | 值 | 说明 |
|------|-----|------|
| model | jev-latest | Jev API 版本 |
| latency | 70-500ms | 端到端响应 |
| cost | $0.042/M input tokens | 输入成本 |
| cost/decision | ~$0.000194 | 单次决策成本 |

### 实验规模
| 参数 | 值 | 说明 |
|------|-----|------|
| episodes | 1000 | 每模式训练集数 |
| log_interval | 10 | 每10集输出日志 |
| save_interval | 100 | 每100集保存检查点 |

## 4. 评估指标

| 指标 | 说明 | 期望对比 |
|------|------|---------|
| completion_rate | 通关率 | Jev-shaped > DQN-only > Jev-only |
| avg_reward | 平均奖励 | Jev-shaped > Jev-router > DQN-only |
| convergence_speed | 首次通关episode数 | Jev-shaped < DQN-only |
| avg_steps | 平均每集步数 | 越少越好（高效通关） |
| jev_calls | Jev总调用次数 | 评估API成本 |
| jev_cost_usd | Jev总成本 | = calls × $0.000194 |
| last_10_avg_reward | 最后10集平均奖励 | 评估训练末期策略质量 |

## 5. 预期结果

### 假设1：Jev Only 表现差但非零
- Jev 能理解"前方有尖刺应该跳"但无法精确控制跳跃时机
- 预期完成率 < 5%，但能通过前几步

### 假设2：Jev Shaped DQN 收敛最快
- Jev 的危险判断帮 DQN 避免大量无意义的死亡探索
- 预期首次通关 episode 从 1000 降至 300-500
- 但 Jev API 调用带来额外延迟和成本

### 假设3：Jev Router DQN 在复杂关卡表现好
- 简单第一关：纯 DQN 已足够，路由开销不划算
- 复杂关卡（需要回溯、多路径选择）：Jev 路由优势显现

### 假设4：成本效益分析
- Jev Shaped DQN 的成本 = episodes × steps × $0.000194
- 1000 episodes × 500 steps × $0.000194 ≈ $97
- 相比训练时间节省，这个成本可接受

## 6. N.E.K.O 集成架构

```
┌─────────────────────────────────────────────┐
│            N.E.K.O Host Process             │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │     CelesteJevRLPlugin (进程)       │    │
│  │                                     │    │
│  │  ┌──────────┐  ┌──────────┐         │    │
│  │  │ JevAgent │  │ DQNAgent │         │    │
│  │  └────┬─────┘  └────┬─────┘         │    │
│  │       │              │              │    │
│  │  ┌────┴──────────────┴─────┐       │    │
│  │  │   ExperimentRunner       │       │    │
│  │  └────────────┬─────────────┘       │    │
│  │               │                     │    │
│  │  ┌────────────┴─────────────┐       │    │
│  │  │     CelesteEnv            │       │    │
│  │  │  (Browser Use / IPC)      │       │    │
│  │  └──────────────────────────┘       │    │
│  │                                     │    │
│  │  LLM Tools: celeste_get_state,      │    │
│  │             celeste_jev_decide     │    │
│  └─────────────────────────────────────┘    │
│                                             │
│  ZMQ IPC ←→ FastAPI Server                  │
└─────────────────────────────────────────────┘
```

### N.E.K.O 提供的能力
- **Plugin SDK**：进程隔离、生命周期管理、IPC通信
- **Browser Use 通道**：通过浏览器自动化操作 Web 版 Celeste Classic
- **LLM Tool Calling**：`celeste_get_state` 和 `celeste_jev_decide` 可在对话中调用
- **五维记忆系统**：记录闯关历史，哪些路径成功/失败
- **Timer**：定时执行实验任务

## 7. 实验流程

```
1. 安装 N.E.K.O + 本插件
2. 配置 Jev API key（config.toml）
3. 准备 Celeste Classic Web 版（或 Everest mod）
4. 选择实验模式
5. 运行实验（run_experiment entry）
6. 定期保存检查点和指标
7. 实验结束后获取结果（get_results entry）
8. 对比四种模式指标
```

## 8. 已知局限

| 局限 | 影响 | 缓解方案 |
|------|------|---------|
| Jev 延迟 70-500ms | 无法做帧级决策(16ms) | Jev 做高层决策，DQN 做底层执行 |
| Jev CJK 准确率低 | 中文状态描述可能误判 | 状态编码使用英文 |
| 蔚蓝需像素级精度 | Jev 无法精确控制时机 | Jev 选策略，DQN 学精确操作 |
| PICO-8 状态读取 | 需要自定义 cartridge | 开发自定义 PICO-8 state export cart |
| API 调用成本 | 长时间训练成本高 | 使用缓存（reward_shaper 已实现） |

## 9. 参考文献

- Madeline's Policy Climb: Climbing Mt. Celeste (Patel, U. Waterloo) — NEAT & DQN on Celeste
- CelesteBot (sc2ad) — NEAT learning for Celeste, C#
- Jev: System One Model (TypeSafe AI, 2026-09) — Structured decision model
- N.E.K.O Plugin SDK Documentation — Plugin development framework
- Sean Goedecke: Two techniques for working with System One models — Goal tiering for Jev game agents
