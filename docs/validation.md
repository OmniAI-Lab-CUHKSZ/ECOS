# Release validation

Release **0.4.0**, 10 November 2026. The update follows the author-supplied
ECOS manuscript and the clarified q_g=1 setting for both environments.

| Executed check | Result |
|---|---|
| Complete CPU pytest suite | **127 passed, 2 skipped, zero failures/errors** |
| RL CLI configuration rendering | All four profiles passed |
| SFT CLI material loading and dry-run | All four profiles passed |
| Python syntax | All 157 source files parsed |
| Packaged materials | Four sets, each200 groups/R1600/S200/natural trajectories1600 |
| Annotation targets/token IDs | Every group compared with retained source; unchanged apart from model-name provenance |
| Natural trajectories | All packaged compressed payloads equal the retained copies |
| Storage/process modules | Six retained modules compared with the previous public commit; unchanged |
| Paper images | Current Figure2, Table1 and FigureD.1 cropped from PDF and visually checked |

The suite ran on Windows/Python3.11 with CPU PyTorch and the retained local
CPU-test dependencies. Two multiprocess CPU FSDP collective tests require
Linux and remain skipped. The one-rank FSDP warning reports NO_SHARD use in
the CPU optimizer test.

The tests exercise the actual Teacher weighting and gradients, opposite-sign
token gaps, shared command-derived multipliers, zero non-command Teacher
gradients, preserved raw R/S gaps, complete-response reduction, configured
group masks, shared Actor/Analyzer optimizer steps, state-centered group
credit, SFT weighting, native storage/resume and framework imports.

Evaluation tests distinguish six-category macro from task-weighted micro,
verify reproducible128/500-goal sampling without replacement, preserve partial
WebShop Score independently of exact Success, and check sample standard
deviations. They use synthetic inputs rather than model-generated benchmark
measurements.

The source tree restores the custom training framework, environment prompts,
memory components and complete four-set material that were missing from the
previous public tree. Annotation model names are corrected from the author's
confirmation; original labels remain in metadata. No dependency was installed
on the shared servers, and server material inspection was read-only through
the existing serialized transport.

GPU training and full environment evaluations were not rerun during this
update. Paper result assets are transcriptions of the supplied PDF. The
release's explicit WebShop sampler does not claim to recover unpublished
historical panel IDs. Ablation and local-GRPO launch recipes were not added.
See [paper_alignment.md](paper_alignment.md) and [../VALIDATION.json](../VALIDATION.json).
