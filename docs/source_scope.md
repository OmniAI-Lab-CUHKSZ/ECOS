# Source scope and curation

This repository supports four full-parameter pipelines: Qwen2.5-3B and 7B on text ALFWorld and WebShop. The entrypoints are `scripts/train_sft.py`, `scripts/train_rl.py`, `scripts/pipeline.py`, model export and the two frozen-policy evaluators.

## Removed inactive experiment paths

| Removed code | Why it was not part of the released runs |
|---|---|
| Root `gigpo/core_gigpo.py` and its trainer import/dispatch | All four profiles use main `adv_estimator=grpo`; the public runner rejects other main estimators |
| GiGPO/GAE/REINFORCE/RLOO/REMAX/PassK advantage branches and GSPO policy-loss branch | The released Actor uses GRPO outcome advantage and vanilla PPO clipping |
| `verl/utils/dataset/rm_dataset.py` | No learned reward model is used; rewards come from the environments |
| `verl/utils/dataset/sft_dataset.py` | R8+S1 SFT is implemented in `response_rs/sft.py` and `sft_data.py` |
| `verl/utils/reflect_lora.py`, adapter loading/export/synchronization and standalone adapter updates | Actor and R/S gradients update the same full model through the same optimizer |
| Overridden generic PPO training loop | The released runtime executes `GRSDReflectRayTrainer.fit` |
| Old backend availability helpers, adapter hijacks and unused configuration fields | They belonged to removed runtime/experiment paths |

Earlier curation also removed SearchQA and other unrelated environments, THOR/vision generation, WebShop baseline/demo programs, static Skill banks, Megatron/SGLang/async backends, external judging, learned Probe/CFI/PGPO paths and Critic/reward-model workers.

Imports, package exports, configurations and runtime callers were updated together. Curation also repaired the version helper's former dependency on a missing shim export. The version helper now reads installed package metadata directly.

## Required shared implementation

| Retained code | Role |
|---|---|
| `verl/trainer/ppo/selective_gigpo.py` | Token-specific Teacher weights, shared command modulation, state grouping and auxiliary advantages; retained optional masking helpers do not import the removed root `gigpo` package |
| `verl/trainer/ppo/rs_sft_protocol.py` | Equations16–20: trajectory-equal state centering and one shared group credit for R8/S1, called by the active trainer |
| `verl/utils/bepo_observation.py` | Frozen-reference observation features for the configured `bepo_unified_obs_action20` auxiliary estimator |
| `verl/trainer/ppo/core_algos.py` | Current GRPO advantage, vanilla policy loss, aggregation and KL utilities |
| FSDP/vLLM workers and sharding managers | Full-model training, generation, exact weight transfer, sleep/wake and RNG boundaries |
| `verl/utils/network_fabric.py` | Caller-selected distributed network setup; renamed from `smoke_fabric.py`, with its implementation preserved |
| Native storage and checkpoint modules | Model/Adam/extra, scheduler/RNG, DataLoader/environment state, readback, fsync and recovery |
| Text environment engines, prompts, history and action masks | Benchmark interaction and exact command-token alignment |
| WebShop Lucene search | Product retrieval inside WebShop; this is part of the shopping environment |

`selective_gigpo` remains an internal configuration/module name for checkpoint and import compatibility. Current main policy advantage is GRPO, while the configured auxiliary observation/action estimator is `bepo_unified_obs_action20`.

The released Teacher objective uses command-only, per-token gaps with coefficient0.01 and a shared command-mean-derived symmetric multiplier. Both environments use q=1, following the author clarification. Joint R/S uses trajectory-equal, state-centered shared group credit with coefficient0.1. SFT and R/S supervise complete Reflection/Skill texts. Full-response main GRPO/KL/entropy masks and the Teacher denominator are retained.

## Verification

Local checks cover imported local symbols, the real GRPO dispatcher, actual full-model weight synchronization/reuse and RNG boundaries, and Actor/R/S gradients. The November update restores the previously omitted framework, prompts, memory modules and material files, aligns the Teacher formula/configuration/evaluators, and preserves the retained storage and process mechanisms. Annotation model names are corrected from author confirmation; target texts and token IDs are preserved. See [validation.md](validation.md) for results and the GPU-validation boundary.
