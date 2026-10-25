# Action-command Teacher supervision with joint Reflection / Skill updates

This source release provides Qwen2.5-3B and 7B SFT material and training pipelines
for ALFWorld and WebShop. See [README.md](README.md) for setup, all four launch
commands, checkpoints, recovery and evaluation.

All four profiles share `configs/rl_common.yaml`:

- Eight natural trajectories per task; all-eight-success groups zero the entire
  Actor Teacher weights and selected masks. Other groups remain eligible.
- Teacher coefficient 0.02, applied only to tokens in the complete action command.
  Reasoning, standalone tags and whitespace tokens receive no Teacher signal.
- The existing `action_command` route uses the summed command gap:
  `delta_action * (1 + sign(delta_action) * tanh(A_action_turn / 0.5))`.
  This signed weight is shared by the command tokens, without an extra sign multiplier.
- Joint Reflection/Skill coefficient 0.1; main loss backward followed by R/S
  backward, with the same Adam step and zero extra optimizer steps.
- Main GRPO, KL and entropy still cover the full valid response; the Teacher term
  retains the original response denominator. The SFT and R/S text objectives stay unchanged.

Four packaged datasets contain 200 tasks each, with eight Reflections and one
Skill per task: 7,200 SFT texts total, plus their natural trajectories, annotation
messages and exact token IDs. SFT uses one task per AdamW update,
`0.5*mean(NLL(R8)) + 0.5*NLL(S)`, three epochs and 600 updates.

The repository contains code and material. Download model weights and environment
assets separately. Portable entrypoints are checked locally; this release does
not claim that all four standardized profiles were rerun on GPUs. See
[docs/validation.md](docs/validation.md) for the exact validation scope and
[THIRD_PARTY.md](THIRD_PARTY.md) for upstream attribution and licenses.
