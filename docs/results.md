# Manuscript figures and results

These assets are reproduced from the working manuscript **Environment-calibrated self-distillation for recursive self-improvement of language agents**, v0.56 (23 October 2026).

| Asset | Manuscript source | Download |
|---|---|---|
| Method overview | Figure 2, `figure_overview_v08.pdf` | [PNG](images/overview.png), [original vector PDF](figures/overview.pdf) |
| Main comparison | Table 1, PDF page 9 | [PNG](images/main-results.png), [CSV](main_results.csv), [LaTeX table body](main_results_table.tex) |
| ALFWorld 3B learning dynamics | Figure 4, `training_alfworld3b_rl150_v56.pdf` | [PNG](images/alfworld-training.png), [original vector PDF](figures/alfworld-training.pdf) |

The PNGs are renders/crops of the original PDFs. Figure data and scientific labels were not redrawn. The CSV transcribes all 24 rows of Table 1 at the manuscript's printed precision.

## Main comparison

The “Ours” rows of manuscript Table 1 are:

| Model | Pick | Look | Clean | Heat | Cool | Pick2 | Seen macro | WebShop Score | WebShop Success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-3B-Instruct | 97.1 | 100.0 | 96.3 | 100.0 | 76.0 | 83.3 | 92.1 | 81.2 | 71.9 |
| Qwen2.5-7B-Instruct | 97.1 | 84.6 | 100.0 | 93.8 | 88.0 | 91.7 | 92.5 | 87.5 | 76.6 |

All values use a 0–100 scale. ALFWorld “Ours” uses all 140 seen tasks and an unweighted average of six category success rates. External comparison values are quoted by the manuscript from [AlignOPSD, Table 1](https://arxiv.org/abs/2609.33391v1) and retain that source's aggregation. An asterisk in the source table denotes skill-conditioned inference. The reported “Ours” policies use the ordinary Actor at inference.

WebShop uses 128 tasks sampled without replacement from goals 0–499 with `RandomState(1000)`. Score is the mean environment score; Success counts exact completions. These columns are not measurements on the full 500-goal panel.

## Unseen ALFWorld

The “Ours” rows of manuscript Table 2 report six-category macro success on all 134 unseen tasks:

| Model | Pick | Look | Clean | Heat | Cool | Pick2 | Macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3B | 83.3 | 100.0 | 80.6 | 87.0 | 81.0 | 82.4 | 85.71 |
| 7B | 75.0 | 88.9 | 83.9 | 82.6 | 90.5 | 100.0 | 86.81 |

Macro averages are distinct from instance-weighted micro success. Main manuscript tables show selected inference runs, not three-seed averages or variability across independent training runs.

## Paper results and the released recipe

The figures and tables preserve the manuscript's historical measurements. The current repository provides a unified recipe for four profiles: **action-command Teacher supervision, coefficient 0.02, all-success Teacher masking, symmetric modulation and joint R/S 0.1**. Existing paper runs include different Teacher scopes and coefficients; the paper numbers are not new measurements of all four unified release profiles.

The paper's historical evaluation protocol uses T0.4 / n1, up to 50 actions with history2 on ALFWorld and 15 actions on WebShop. The portable evaluators here default to max30 / history5 and evaluate full ALFWorld 274 or WebShop 500 goals. Reproducing a paper table requires its corresponding checkpoint, inference run selection and historical protocol.

This curation ran local objective, dependency, storage and entrypoint checks. It did not start new GPU training or benchmark evaluation. [Validation details](validation.md).
