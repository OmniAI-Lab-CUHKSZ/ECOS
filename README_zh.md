# Environment-calibrated self-distillation

Qwen2.5-3B / 7B 在 ALFWorld、WebShop 上的 SFT 材料、全参数 SFT 代码和 RL 流水线。
四个入口统一为：**全成功组屏蔽 Teacher、Teacher 系数 0.02、只监督 action 命令 token、对称调制、joint R/S 系数 0.1**。

## 内容

| 入口 | SFT 材料 | RL 配置 |
|---|---|---|
| ALFWorld 3B | `data/sft/alfworld_3b` | `configs/alfworld_3b.yaml` |
| ALFWorld 7B | `data/sft/alfworld_7b` | `configs/alfworld_7b.yaml` |
| WebShop 3B | `data/sft/webshop_3b` | `configs/webshop_3b.yaml` |
| WebShop 7B | `data/sft/webshop_7b` | `configs/webshop_7b.yaml` |

每套材料是 200 任务 × (8 Reflection + 1 Skill)，共 1,800 条文本；四套共 7,200 条。
包含对应的自然轨迹、标注提示、原始 token IDs。权重和环境安装数据单独下载。

`verl/`、`agent_system/` 是框架源码；`response_rs/` 和 `scripts/` 提供可移植入口。
方法配置集中在 `configs/rl_common.yaml`，四套入口仅覆盖环境、模型尺寸和资源预算。
Teacher监督范围由 `algorithm.selective_gigpo.actor_signal_scope=action_command` 控制；`response_only_logits` 表示logits计算优化。

## 安装与环境

训练使用 Linux、Python 3.11、CUDA 和 2–8 张 GPU。Windows 可读取材料、渲染配置和运行适用的 CPU 测试。
单机启动器管理独立 Ray、归档服务及训练进程。多机底层使用框架的 Ray 资源池；本仓库不包含特定集群的 SSH 调度、地址或账号。

```bash
python -m venv .venv
source .venv/bin/activate
bash scripts/setup.sh
python scripts/download_model.py --size 3b --output models/Qwen2.5-3B-Instruct
python scripts/download_model.py --size 7b --output models/Qwen2.5-7B-Instruct
```

WebShop 需要 Java 11。环境依赖和下载方式见 [docs/environments.md](docs/environments.md)。
设置 `ALFWORLD_DATA` 为完整 ALFWorld 数据目录，`WEBSHOP_ASSET_ROOT` 为包含 `data/`、`search_engine/` 的 WebShop 资产目录。
预留全参数 Model/Adam 检查点的磁盘和主机内存：7B 的 FP32 Model/Adam 单个全状态约 90GB；运行时归档保留 80GB 余量。
按默认保留各 SFT epoch 与 RL 每20步里程碑，SFT→RL 整套建议准备约0.6TB（3B）或1.3TB以上（7B）的可用磁盘。

## 1. SFT

也可用一个入口顺序执行 SFT600 → RL150。例如：

```bash
python scripts/pipeline.py --config webshop_7b --base-model models/Qwen2.5-7B-Instruct --gpus 0,1,2,3,4,5,6,7 --webshop-assets "$WEBSHOP_ASSET_ROOT" --run-dir runs/webshop_7b
```

`--config` 可选上表四套，ALFWorld 传 `--alfworld-data`。`--stage sft/rl` 可单独推进某一阶段，`--dry-run` 打印顺序且不启动作业。

```bash
python scripts/train_sft.py --config alfworld_3b --model models/Qwen2.5-3B-Instruct --gpus 0,1,2,3,4,5,6,7 --output runs/sft_alfworld_3b
python scripts/train_sft.py --config alfworld_7b --model models/Qwen2.5-7B-Instruct --gpus 0,1,2,3,4,5,6,7 --output runs/sft_alfworld_7b
python scripts/train_sft.py --config webshop_3b --model models/Qwen2.5-3B-Instruct --gpus 0,1,2,3,4,5,6,7 --output runs/sft_webshop_3b
python scripts/train_sft.py --config webshop_7b --model models/Qwen2.5-7B-Instruct --gpus 0,1,2,3,4,5,6,7 --output runs/sft_webshop_7b
```

每个 Adam 更新只使用一个任务的九条文本，损失为 `0.5 × mean(NLL(R1..R8)) + 0.5 × NLL(S)`。
仅 response 位置计平均 NLL；全参数 AdamW、LR 5e-6、3 epochs / 600 更新、constant LR、warmup 0、weight decay 0、clip 1。
使用 FP32 master 和 Adam、BF16 前向。32768 上限，超长报错；跨 rank 的占位样本权重为零。
每个 epoch 保存可直接用于 RL 的 HF 模型，以及 FP32 Model / Adam / RNG / 数据位置全状态。

`--dry-run` 只读取材料并打印启动参数。`--smoke` 明确启动两步真实 GPU 训练。
SFT 续训使用 `--resume-from runs/.../checkpointNNN` 和一个新的 `--output`，恢复原 rank 数和完整状态。

## 2. 统一 RL

以下四个命令的方法设置相同。模型目录指向对应 SFT 的 `checkpoint600`。

```bash
python scripts/train_rl.py --config alfworld_3b --model runs/sft_alfworld_3b/checkpoint600 --gpus 0,1,2,3,4,5,6,7 --alfworld-data "$ALFWORLD_DATA" --run-dir runs/rl_alfworld_3b
python scripts/train_rl.py --config alfworld_7b --model runs/sft_alfworld_7b/checkpoint600 --gpus 0,1,2,3,4,5,6,7 --alfworld-data "$ALFWORLD_DATA" --run-dir runs/rl_alfworld_7b
python scripts/train_rl.py --config webshop_3b --model runs/sft_webshop_3b/checkpoint600 --gpus 0,1,2,3,4,5,6,7 --webshop-assets "$WEBSHOP_ASSET_ROOT" --run-dir runs/rl_webshop_3b
python scripts/train_rl.py --config webshop_7b --model runs/sft_webshop_7b/checkpoint600 --gpus 0,1,2,3,4,5,6,7 --webshop-assets "$WEBSHOP_ASSET_ROOT" --run-dir runs/rl_webshop_7b
```

追加 `--background` 可后台运行；`--status --run-dir ...` 查看当前进度。
失败后先保留该 attempt，再以相同命令追加 `--resume`，从实时最新完整检查点恢复，下一步为 saved+1。
已有活跃进程或退出状态不明时，启动器不会重复提交。普通续训保持物理 rank 数；改变拓扑需要框架的完整原生转换路径，不能只加载模型重建 Adam。

150 外层步、每步 16 任务 × G8、logical world 8 / mini256、主 token 预算 8608 / micro32、LR 1e-6、warmup 0、clip 0.2、KL 0.01、entropy 0.001。
交互 max30 / history5，prompt2048 / response512；每步全状态保存，每 5 步及最终执行固定 128 验证。
R/S prompt 上限 32256，原生上下文 32768，R/S 预算使用 `max(8608, padded_width)`，不静默截断 R/S。
3B/7B 的 vLLM 显存预算分别为 0.24/0.30，可用 `--vllm-memory-utilization` 按硬件调整，不改变训练数学。

## 方法

使用现有 `action_command` 路径。令 `delta_action` 为本回合完整 action 命令各 token 的 Teacher / old Student log-probability 差值之和，同回合 action advantage 为 `A_action_turn`：

```text
weight_t = action_mask_t × delta_action × [1 + sign(delta_action) × tanh(A_action_turn / 0.5)]
Teacher outer coefficient = 0.02
```

Teacher 只作用于 `<action>...</action>` 内命令文字对应的原始 token，同一命令共享上述权重。think、独立标签和纯空白 token 不接收 Teacher 信号；仍遵守有效 response、完整命令及原始 token 边界的 action mask 检查。
不再额外乘 action gap 的符号。Teacher、old Student、权重和 advantage 均停止梯度；实际 Teacher 项使用保留的 signed PPO clipping。
主 GRPO / KL / entropy 继续覆盖完整有效 response，Teacher 项保留原始 response 分母。SFT 和 R/S 的 Reflection / Skill 目标文本监督保持原配方。

同题八条自然轨迹的**环境成功结局均为成功**时，整组 Actor Teacher 权重及 selected mask 清零。
padding、副本和重复 turn 不增加轨迹数；7/8 成功仍保留 Teacher。
屏蔽仅作用于 Actor Teacher，主 GRPO / KL / entropy、完整 response 分母和 R/S 不变。

R/S 使用 `A_step × raw action command gap`：Reflection 对轨迹有效 turn 取均值，Skill 对组内轨迹等权取均值，再各自 clip 到 ±3。
R/S 系数 0.1、`seq-mean-token-mean`；主 loss 反向后累加 R/S 梯度，同一次 Adam，额外 optimizer 步为 0。
线上 Reflection、Skill 和 Teacher 都来自策略本身，无外部 GPT、judge、Probe/CFI 或测试时更新。

## 保存和运行材料

检查点包含各 rank Model / Adam / scheduler / RNG、DataLoader 和环境状态，以及方法观测。
原生 shard 直接写入本次独立归档服务，PUT / fsync / GET 逐字节回读；`DIRECT_SAVED` 是指针回执。
只有完整保存后才轮换旧状态，保留最新点和每 20 步里程碑。原始 rollout、验证和运行日志随检查点保留。
服务只接受本次来源和令牌；退出时按实际进程身份退休本次训练与私有 Ray，不执行全局 `ray stop`。
vLLM profile/sleep 的两处非 OOM 显存计量断言沿用当前实现改为提示；实际预算计算、卸载和 CUDA OOM 传播保持。

## 验证与来源

冻结 Actor 导出与评测（3B/7B 共用）：

```bash
python scripts/export_model.py --checkpoint runs/rl_alfworld_3b/checkpoints/global_step_150 --output models/alfworld_3b_rl150
python scripts/evaluate_alfworld.py --model models/alfworld_3b_rl150 --alfworld-data "$ALFWORLD_DATA" --gpu 0 --seeds 0,1,2 --output runs/eval_alfworld_3b
python scripts/export_model.py --checkpoint runs/rl_webshop_3b/checkpoints/global_step_150 --output models/webshop_3b_rl150
python scripts/evaluate_webshop.py --model models/webshop_3b_rl150 --asset-root "$WEBSHOP_ASSET_ROOT" --gpu 0 --seeds 0,1,2 --output runs/eval_webshop_3b
```

ALFWorld 为 Seen140 + Unseen134，WebShop 为固定完整500目标；保存每题轨迹、环境成功结局、每种子计数和均值±样本标准差。
评测 T0.4 / top_p1 / top_k-1 / n1 / max30 / history5 / prompt2048 / response512，无 Skill/Teacher 辅助与测试时更新。
相同目录可续接未完成题目，完整结果会直接读取。

```bash
python -m pytest tests -q
python scripts/inspect_material.py
python scripts/train_rl.py --config webshop_7b --model models/webshop_7b_sft600 --gpus 0,1,2,3,4,5,6,7 --dry-run
```

本次整理的验证范围见 [docs/validation.md](docs/validation.md)。全成功屏蔽、调制及 joint R/S 沿用保留的实现；本次 action-only / 0.02 的四套入口经过本地检查，不表示已经重新完成 GPU 训练，也不据此声称方法收益。
上游来源与许可证见 [THIRD_PARTY.md](THIRD_PARTY.md)。
