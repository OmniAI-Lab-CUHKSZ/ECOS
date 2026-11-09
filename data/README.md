# SFT material

Four datasets are included under `sft/{alfworld,webshop}_{3b,7b}`. Each has:

- 200 task groups; eight Reflections and one Skill per group (1,800 target texts).
- `groups-*.jsonl.gz`: exact retained prompt IDs, response IDs, target text,
  annotation messages, success labels and task identifiers.
- `natural/trajectories-*.jsonl.gz`: the corresponding eight natural trajectories.
- `tasks.json`: task order and portable environment identifiers.
- `metadata.json`: tokenizer family, annotation model, counts and length statistics.

All four sets contain 1,600 natural trajectories each (6,400 total), in addition to 7,200 SFT target texts. The annotation model is GPT-6.1 sol with maximum reasoning effort. On 10 November 2026 the author confirmed that the historical ALFWorld3B label `gpt-5.6-terra` and WebShop3B label `gpt-6-sol` identify this model. Those labels remain in `recorded_annotator` and `annotation_model_provenance`; annotation text and token IDs were not regenerated. Both SFT sizes initialize from the original Qwen2.5-Instruct checkpoint; WebShop collection uses GRPO40 policies.

All 200 groups are used by the supplied SFT recipe. Historical `sft_split` labels
inside the material do not create a held-out split for this 600-update recipe.
ALFWorld game paths are relative to `ALFWORLD_DATA/json_2.1.1`. Token IDs and
message/target text are retained; use the corresponding Qwen2.5 tokenizer rather
than silently retokenizing with another vocabulary.

The material contains model-generated annotations and benchmark observations;
success labels describe the recorded environment outcome. An annotation is not
an independent proof that an action was necessary or caused success. Benchmark
and product content retain their upstream rights. The code LICENSE does not
grant rights over the underlying ALFRED/WebShop data.

`rl/*/{train,test}.parquet` contain 16/128 empty prompt records for batching. The
live environment supplies observations and prompts; these are not SFT examples
or a list of all evaluation instances.
