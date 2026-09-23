# 项目交接包 — Celeste Jev RL

> **N.E.K.O 插件：基于 Jev 决策模型 + DQN 强化学习的蔚蓝闯关实验**
>
> 交接日期：2026-09-23
> 仓库地址：https://github.com/acekira3-afk/n.e.k.o_plugin_celeste_jev_rl
> 代码行数：4,327 行 Python + 581 行数据提取工具
> Git 提交：4 次
> 单元测试：36 个（全部通过）

---

## 1. 项目概述

### 1.1 目标

将 Jev（TypeSafe AI 的纯合成数据决策模型）与 N.E.K.O（网络型情感知性生命体 Agent 框架）和 DQN 深度强化学习结合，用于蔚蓝（Celeste）游戏的闯关学习实验。

核心创新点：**Think-Execute 三段式架构**——将 Jev 的 70-500ms 延迟从"缺陷"转化为"角色思考时间"。

### 1.2 架构

```
┌─ 感知阶段（1-3帧）──────────────────┐
│  读取游戏状态：位置、速度、障碍物、  │
│  体力、冲刺可用性                    │
└──────────┬─────────────────────────┘
           ▼
┌─ 思考阶段（Jev + N.E.K.O）──────────┐
│  1. Jev 判断：当前是什么"情境"       │
│  2. Jev 选择：对应的操作脚本 ID      │
│  3. N.E.K.O 表现：                   │
│     - Live2D 表情变化（9种）         │
│     - 独白文字（思考/执行/成功/失败）│
│     - 位置记忆（历史失败/成功记录）  │
│  持续 0.3-1 秒（拟人思考时间）       │
└──────────┬─────────────────────────┘
           ▼
┌─ 执行阶段（帧精确脚本）─────────────┐
│  按选定脚本逐帧执行输入序列          │
│  脚本是预录制的帧精确操作表          │
│  执行期间不调用 Jev                  │
│  直到脚本结束 → 回到感知阶段         │
│  支持 5 种中断条件检测               │
└─────────────────────────────────────┘
```

### 1.3 四种实验模式

| 模式 | 说明 |
|------|------|
| `jev_only` | 纯 Jev 零训练决策（基线上限） |
| `dqn_only` | 纯 DQN 标准 RL（基线） |
| `jev_shaped_dqn` | DQN + Jev reward shaping（核心实验） |
| `jev_router_dqn` | Jev 高层路由 + DQN 底层执行 |

---

## 2. 文件清单

### 2.1 核心代码（core/）

| 文件 | 行数 | 职责 |
|------|------|------|
| `__init__.py` | 293 | 插件主入口，4 个 plugin_entry + 2 个 LLM tool |
| `state_encoder.py` | 205 | 蔚蓝 26 维状态 → Jev 文本编码 |
| `jev_agent.py` | 333 | Jev 决策层（4 层目标层级 + reward shaping 评估） |
| `dqn_agent.py` | 199 | DQN 训练层（4 层网络 + 回放 + 目标网络） |
| `reward_shaper.py` | 98 | Jev 奖励塑形器（带缓存） |
| `script_library.py` | 316 | 10 个内置帧精确脚本 |
| `script_executor.py` | 239 | 帧精确回放 + 5 种中断条件检测 |
| `personality.py` | 337 | 9 种 Live2D 表情 + 独白 + 位置记忆 |
| `celeste_env.py` | 205 | 蔚蓝环境（PICO-8 / Everest mod 双后端） |
| `experiment_runner.py` | 273 | 实验调度器（4 模式 + Think-Execute 循环） |

### 2.2 测试（tests/）

| 文件 | 行数 | 职责 |
|------|------|------|
| `test_plugin.py` | 384 | 36 个单元测试（全部通过） |
| `test_first_data.py` | 389 | 50 集 mock 环境数据测试 |
| `test_real_data.py` | 440 | 100 集真实游戏数据训练测试 |

### 2.3 工具（tools/）

| 文件 | 行数 | 职责 |
|------|------|------|
| `extract_game_data.py` | 581 | .bin 地图解析 + .celeste 存档解析 + 训练样本生成 |

### 2.4 数据（data/training/）

| 文件 | 大小 | 内容 |
|------|------|------|
| `training_samples.json` | ~100KB | 259 个训练样本（232 关卡级 + 27 地图级） |
| `map_data.json` | ~80KB | 27 个 .bin 地图的完整解析数据 |
| `save_data.json` | ~30KB | 3 个存档的完整解析数据 |
| `summary.json` | ~5KB | 数据提取摘要（实体/情境/难度分布） |

### 2.5 文档（docs/）

| 文件 | 内容 |
|------|------|
| `experiment_design.md` | 详细实验方案（8 节） |
| `dev_log.md` | 开发日志（9/22-9/23） |
| `first_data_test_report.json` | 50 集 mock 测试报告 |
| `extended_data_test_report.json` | 100 集 mock 测试报告 |
| `real_data_training_report.json` | 100 集真实数据训练报告 |

### 2.6 配置 & CI/CD

| 文件 | 内容 |
|------|------|
| `plugin.toml` | N.E.K.O 插件清单 |
| `config.example.toml` | 配置模板 |
| `.github/workflows/verify.yml` | CI：测试 + 结构检查 |
| `.github/workflows/release.yml` | CD：标签触发 .neko-plugin 发布 |

---

## 3. 数据测试结果汇总

### 3.1 三次测试对比

| 指标 | Mock 50 集 | Mock 100 集 | 真实数据 100 集 |
|------|-----------|------------|----------------|
| 脚本执行数 | 154 | 307 | 187 |
| 成功率 | 34.2% | 34.0% | 14.7% |
| 中断率 | 27.3% | 27.0% | 25.7% |
| 总思维数 | 159 | 113 | 157 |
| 位置记忆 | 27 | 41 | 15 |
| 总帧数 | 1,877 | 3,676 | 1,904 |
| 耗时 | 69.9s | 139.2s | 82.8s |

### 3.2 各脚本成功率

| 脚本 | Mock 100 集 | 真实数据 100 集 |
|------|------------|----------------|
| simple_move_right | 100% | 100% |
| hyper_jump | 100% | 100% |
| dash_across_gap | 55% | 33% |
| wave_dash_chain | 0% | 0% |

### 3.3 真实游戏数据提取

- **27 个 .bin 地图文件**解析成功
- **232 个关卡**（level names 提取）
- **489 个实体**（15+ 类型分类）
- **3 个存档**（8,468 总死亡，27,321 总冲刺，118 草莓）
- **13 种情境分类**：spinner_field / spring_bounce / touch_switch_puzzle / spike_corridor / dash_across_gap / crumble_platform / move_block_puzzle / booster_section / strawberry_hunt / swap_block_puzzle / checkpoint_run / simple_move / lightning_dodge

### 3.4 关键发现

1. **架构稳定**：100 集无崩溃，数据一致性好
2. **真实数据更难**：成功率从 34% 降到 14.7%，因为使用了真实难度分级的死亡概率
3. **简单脚本可靠**：simple_move_right 和 hyper_jump 在两种环境下都 100%
4. **wave_dash_chain 需要调整**：0% 成功率说明脚本帧数据有问题
5. **中断检测有效**：25-27% 脚本被状态偏移打断
6. **位置记忆积累**：15-41 个位置记住了历史结果
7. **人格表情循环正确**：thinking → focused/panicked → happy/frustrated

---

## 4. 环境配置

### 4.1 依赖

```bash
pip install pyautogui Pillow numpy
# Jev API（可选，mock 模式不需要）
# N.E.K.O 插件 SDK
```

### 4.2 游戏路径

| 平台 | 路径 |
|------|------|
| macOS Steam 版 | `~/Library/Application Support/Steam/steamapps/common/Celeste/Celeste.app/Contents/Resources/Content/Maps/` |
| macOS 存档 | `~/Library/Application Support/Celeste/Saves/` |
| CrossOver 存档 | `~/Library/Application Support/CrossOver/Bottles/Steam/drive_c/Program Files (x86)/Steam/steamapps/common/Celeste/Saves/` |

### 4.3 运行命令

```bash
# 单元测试
python3 -m pytest tests/ -v

# 数据提取
python3 tools/extract_game_data.py

# Mock 数据测试（50 集）
python3 tests/test_first_data.py

# 真实数据训练测试（100 集）
python3 tests/test_real_data.py
```

---

## 5. 待完成事项

### 5.1 高优先级

| # | 事项 | 说明 |
|---|------|------|
| 1 | **修复 wave_dash_chain 脚本** | 0% 成功率，帧数据需要重新录制 |
| 2 | **真实游戏输入映射** | pyautogui 的 Enter/Space/C 键无法确认蔚蓝菜单，需要排查 CrossOver/Wine 键盘映射 |
| 3 | **屏幕区域识别** | 用 OCR/像素匹配从截图提取真实游戏状态（位置/速度/障碍物） |
| 4 | **接入真实 Jev API** | 当前使用 MockJevAgent，需要替换为实际 API 调用 |

### 5.2 中优先级

| # | 事项 | 说明 |
|---|------|------|
| 5 | 脚本录制工具 | 编写交互式工具录制玩家操作为帧精确脚本 |
| 6 | DQN 训练管线 | 完成 jev_shaped_dqn 模式的端到端训练 |
| 7 | N.E.K.O Live2D 集成 | 将人格表情层对接实际 Live2D 模型 |
| 8 | Everest mod 安装 | 安装 Everest mod 以通过 Lua API 直接读取游戏内存 |

### 5.3 低优先级

| # | 事项 | 说明 |
|---|------|------|
| 9 | CJK 状态编码优化 | Jev 中文准确率偏低，状态描述用英文 |
| 10 | 脚本库扩展 | 增加 wall_jump_climb、spike_corridor_wait 等脚本 |
| 11 | 多关卡脚本适配 | 当前脚本通用，需要按关卡特征选择不同脚本变体 |
| 12 | N.E.K.O 插件市场提交 | CI 通过后在 market.project-neko.cn 提交审核 |

---

## 6. 技术决策记录

| 决策 | 选择 | 原因 |
|------|------|------|
| Jev 延迟处理 | Think-Execute 三段式 | 500ms 延迟变成"思考时间"特性而非缺陷 |
| 状态表示 | 26 维数值 → 文本编码 | Jev 是文本输入模型，需要语义化状态 |
| 动作选择 | 脚本库（非单帧动作） | 蔚蓝需要帧精确操作，Jev 无法做帧级决策 |
| 人格表现 | N.E.K.O PersonalityLayer | 思考过程外显为表情/独白，从工具变角色 |
| 数据提取 | 二进制字符串扫描 | .bin 格式无公开文档，字符串扫描最可靠 |
| 死亡概率 | 基于真实存档数据 | easy 1% / medium 3% / hard 6% + 高死亡区域加权 |

---

## 7. 联系信息

- **GitHub 仓库**：https://github.com/acekira3-afk/n.e.k.o_plugin_celeste_jev_rl
- **N.E.K.O 插件市场**：https://market.project-neko.cn
- **Jev API**：https://jev.ai
- **蔚蓝速通社区**：https://celesteclassic.github.io

---

## 8. 许可证

MIT License — 见 [LICENSE](LICENSE) 文件。
