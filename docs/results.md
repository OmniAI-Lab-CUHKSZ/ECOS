# ECOS manuscript results and evaluation protocol

These values and PNGs come from **ECOS: Environment-Calibrated On-Policy
Self-Distillation for Language Agents**, the 25-page manuscript supplied by
the author on10 November2026. Updating the implementation does not create new
training or evaluation measurements.

| Asset | Manuscript source | Repository file |
|---|---|---|
| Method overview | Figure2, page4 | [PNG](images/overview.png) |
| Main comparison | Table1, page8 | [PNG](images/main-results.png), [CSV](main_results.csv), [LaTeX](main_results_table.tex) |
| 3B learning dynamics | FigureD.1, page24 | [PNG](images/alfworld-training.png) |

The PNGs are cropped renders of the supplied PDF; plotted data and scientific
labels were not redrawn. The CSV retains quoted comparison rows and includes
separate standard-deviation columns. Empty deviation cells mean the PDF did
not print a deviation; they are not imputed zeros.

## Main comparison

All entries use a0–100 scale. Ours reports mean ± sample standard deviation
over three evaluation seeds for each trained checkpoint.

| Model | Pick | Look | Clean | Heat | Cool | Pick2 | Seen macro | WebShop Score | WebShop Success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3B |99.0±1.6|82.1±4.4|95.1±2.1|97.9±3.6|82.7±10.1|86.1±6.4|90.5±1.0|85.5±2.6|75.5±1.9|
| 7B |100.0|94.9±4.4|100.0|95.8±3.6|81.3±2.3|87.5±4.2|93.3±0.9|88.3±1.3|80.5±1.4|

ALFWorld uses all140 seen tasks and averages the six category success rates
equally. WebShop uses128 goals sampled without replacement per evaluation
seed from goals0–499; methods with the same seed share the panel. Catalogue
seed is0, with the training-compatible1,000-product search environment.
Success counts exact completions; Score is mean normalized final environment
reward multiplied by100. Quoted methods retain the source aggregation in
AlignOPSD Table1. An asterisk denotes skill-conditioned inference in that
source; our Actor receives ordinary interaction history at evaluation.

## Unseen ALFWorld

Table2 uses all134 unseen tasks and the same six-category macro definition.

| Model | Pick | Look | Clean | Heat | Cool | Pick2 | Macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3B |80.6±2.4|92.6±3.2|86.0±4.9|91.3±4.3|84.1±9.9|80.4±9.0|85.8±1.8|
| 7B |76.4±6.4|94.4|82.8±1.9|85.5±5.0|92.1±2.7|88.2|86.6±1.2|

The three seeds measure evaluation variability, not independent training
seeds. Main task results use the final150-update checkpoint.

## Executable evaluation

The public evaluators use temperature0.4/top-p1/top-k−1/n1. ALFWorld uses
max30/history5/prompt2048; WebShop uses max15/history2/prompt4096. Both use
response512. Per-task trajectories, per-seed summaries, WebShop panel IDs and
sample standard deviations are retained. ALFWorld also reports micro success
for reference; it is not the paper's main metric.

WebShop panels use `random.Random(seed).sample(range(500),128)`. This is the
release's explicit reproducible sampler. Historical paper panel IDs were not
supplied, so newly generated IDs are not described as recovered paper panels.

The author clarified that q_g=1 in both tasks and confirmed GPT-6.1 sol/max
for all offline annotations. Those clarifications are recorded in the method
and material metadata. This update adds no ablation or local-GRPO launch
recipes. [Validation scope](validation.md).
