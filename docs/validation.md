# Release validation

Release **0.3.0**, curated on 25 October 2026. Local validation used Windows, Python 3.11 and CPU PyTorch. The target GPU runtime is listed separately in `requirements/runtime.txt`.

| Check | Result |
|---|---|
| Unit tests and actual Actor/R/S loss methods | 121 passed, 2 platform skips |
| Four Hydra profiles | Composed and resolved successfully |
| SFT, RL and complete pipeline dry-runs | 12/12 exit 0; no training outputs created |
| Retained Python sources | Compiled successfully; local import targets present |
| SFT and RL material files | All 94 files match the supplied v0.2.0 ZIP byte for byte |
| SFT coverage | Four sets, each with 200 tasks, 1,600 Reflections, 200 Skills and 1,600 trajectories |
| Paper assets | Original vector figures rendered and visually inspected; Table 1 cropped from the PDF and 24 rows transcribed to CSV |
| Source packaging | Git ignore rules preserve required framework/simulator source and exclude weights |

The tests exercise:

- 8/8-success masking of Actor Teacher weights and selected masks; 7/8 remains eligible. Padding and repeated turns do not create new trajectories.
- Action-command spans and signed command-gap modulation, including zero Teacher gradient on think text, tags, whitespace-only tokens and padding.
- The actual Actor loss with Teacher coefficient 0.02, signed PPO clipping and the unchanged response denominator.
- Main and R/S gradient accumulation, coefficient 0.1, a shared optimizer update and zero additional optimizer steps.
- Reflection turn means, equally weighted Skill trajectory means, clip ±3, signed and zero rewards.
- Actual trainer NPZ emission and action-signal validation by the native update observer.
- Full-text SFT response NLL, R8/S1 weighting, and zero-weight padding across 2–8 ranks.
- Refactored trainer initialization, required context checks, Teacher scoring with temporary inference padding, and retained TextWorld configuration fields.
- Real GRPO dispatch for all four profiles, local imported symbols, full-model weight synchronization/reuse and RNG boundaries.
- Storage snapshots and exclusion of in-progress archive files or duplicate native payloads.

Two multi-process CPU FSDP collective tests are skipped on Windows and retained for Linux. Streamed AdamW CPU parameter/state equivalence was verified in the earlier package; this curation did not change its implementation.

The cleanup removed unused environments, visual/THOR generation, website/baseline demos, static Skill banks, old launchers, Megatron/SGLang/async backends, learned Probe/CFI/PGPO paths, external judging and unused Critic/reward-model workers. Required action masks and frozen-reference observation features remain in use. Native Model/Adam/extra, scheduler/RNG, DataLoader/environment recovery and existing process/storage measures remain included.

The further runtime audit removed the unused root GiGPO implementation, generic reward-model/SFT datasets, standalone LoRA paths, overridden PPO loop and inactive advantage/policy-loss branches. The 19 compared active loss/state implementations remain unchanged, and all 94 material files match the supplied package byte for byte. Details and retained dependencies are listed in [source_scope.md](source_scope.md).

**No new GPU SFT, RL or benchmark evaluation was started for this release.** Local tests and dry-runs do not establish Linux/NCCL runtime correctness or new benchmark scores. Manuscript figures and results are historical measurements described in [results.md](results.md).
