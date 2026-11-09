# ECOS: Environment-Calibrated On-Policy Self-Distillation for Language Agents

See [README.md](README.md) for installation, the four full-method profiles,
SFT600, RL150, recovery, export and three-seed frozen evaluation.

- Teacher coefficient 0.01; command-content tokens only.
- A shared multiplier uses the command's mean gap and environmental advantage.
  Each selected token retains its own Teacher-Actor gap; the raw summed command
  gap remains available for R/S credit.
- Both environments use q=1: all-success groups retain Teacher supervision,
  following the author's clarification on 10 November 2026.
- R/S uses trajectory-equal, within-state centered group credit, clipped to
  [-3,3] and shared by R8 and S1. Reflection and Skill losses each receive half
  the Analyzer weight. Coefficient 0.1, one shared optimizer, zero extra steps.
- Main GRPO/KL/entropy and the Teacher token-mean denominator cover the complete
  valid response. Actor Teacher supervision excludes reasoning and delimiters.

Each of the four datasets contains 200 groups, 1,600 natural trajectories,
1,600 Reflections and 200 Skills. All annotations use GPT-6.1 sol / max, as
confirmed by the author; original recorded model labels remain in metadata.
SFT starts from the original Qwen2.5-Instruct backbones. WebShop collection uses
the corresponding GRPO40 policies.

ALFWorld uses mini256, max30/history5, Actor prompt2048, Reflection8096 and
Teacher4096. WebShop uses mini64, max15/history2, Actor prompt4096,
Reflection16384, Teacher6144 and catalogue seed0. Response length is512.

Evaluation reports six-category macro success on ALFWorld Seen140/Unseen134;
WebShop uses128 goals per seed and reports Success plus Score. The default
three evaluation seeds are0,1,2, with sample standard deviations. Paper tables
are transcriptions, not new results from this update. Ablation and local-GRPO
launch recipes are outside this release update.

See [docs/validation.md](docs/validation.md) for checks and limitations and
[THIRD_PARTY.md](THIRD_PARTY.md) for upstream attribution and licenses.
