# Alignment with the current ECOS manuscript

This update follows the 25-page manuscript supplied by the author on
10 November 2026. It provides the four full-method profiles for
ALFWorld/WebShop and Qwen2.5-3B/7B.

| Paper component | Implementation |
|---|---|
| Equations 11–13, summed/mean command gaps and token-specific weights | `verl/trainer/ppo/selective_gigpo.py::token_route`; `response_rs/teacher_signals.py` checks saved natural signals |
| Equations 8–10, empirical Bellman/GAE command advantage | `observation_state_keys` and `bepo_action_advantages` in the same module |
| Equations 16–20, state-centered shared group credit | `verl/trainer/ppo/rs_sft_protocol.py::centered_credit`, called by `GRSDReflectRayTrainer` |
| Shared Actor/Analyzer optimizer and complete-response denominator | `verl/workers/actor/dp_actor.py` and the joint-update tests |
| Table B.2, task-specific budgets | `configs/rl_common.yaml` and the four task/size overlays |
| Appendix C.2, macro success and WebShop Success/Score | `response_rs/alfworld_evaluation.py` and `webshop_evaluation.py` |

The author clarified two facts during this alignment:

- Both environments use **q_g=1**. All-success groups retain Teacher
  supervision; the configuration sets `skip_teacher_on_all_success=false`.
- All four offline annotation sets use **GPT-6.1 sol / maximum reasoning
  effort**. Historical ALFWorld3B and WebShop3B model labels are corrected
  from author confirmation and preserved separately in the material metadata.

Both model sizes initialize SFT from original Qwen2.5-Instruct weights.
WebShop trajectory collection uses the corresponding GRPO40 policy.
Each packaged set contains 200 groups, 1,600 natural trajectories,
1,600 Reflections and 200 Skills. Saved token IDs and target texts are retained.

WebShop panel generation is explicitly defined as
`random.Random(seed).sample(range(500),128)`. The saved IDs support paired
comparisons across methods. The manuscript does not provide historical panel
IDs; this sampler is not presented as recovery of those IDs.

The update restores omitted framework and material files and fixes native
export filenames to use the actual saved world size. Existing run locks,
complete-state checkpointing, durable fsync/readback, resource protection and
precise process ownership remain included. It adds no ablation or local-GRPO
launch recipes. Paper table values are transcriptions; GPU training and full
environment evaluation were not rerun for this code update.

See [validation.md](validation.md) for executable checks and their scope.
