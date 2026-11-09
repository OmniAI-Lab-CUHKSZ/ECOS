"""Save existing natural-rollout signals, without inference or RNG consumption."""
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch


def _array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()  # NumPy cannot encode bfloat16; conversion is exact.
        return value.numpy()
    return np.asarray(value)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return _jsonable(_array(value).tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    with Path(path).open('w', encoding='utf-8') as stream:
        json.dump(_jsonable(value), stream, ensure_ascii=False, allow_nan=False)


def save_analysis_step(run_dir, step, *, batch, rows, advantage,
                       observation_embeddings, command_gap, entropy,
                       actor_eligible_mask, reflect_out, prior_out,
                       rs_train_batch, reflect_context, tokenizer, config):
    """Snapshot all batch rows (including recorded padding), before updates.

    Stored coefficients are not parameter-space gradients. A different model
    forward/layer requires a compatible checkpoint; this archive does not contain
    model weights or full-vocabulary distributions.
    """
    started = time.monotonic()
    root = Path(run_dir) / 'analysis_assets'
    root.mkdir(parents=True, exist_ok=True)
    # Keep retries as separate attempts; never overwrite an earlier observation.
    target = Path(tempfile.mkdtemp(prefix=f'step_{int(step):03d}_', dir=root))
    tensors = batch.batch
    count = len(rows)
    if count != len(tensors['responses']):
        raise ValueError('Analysis row order does not match training tensors')

    tensor_keys = (
        'input_ids', 'prompts', 'responses', 'attention_mask', 'position_ids',
        'response_mask', 'loss_mask', 'old_log_probs', 'teacher_log_probs',
        'ref_log_prob', 'advantages', 'returns', 'token_level_scores',
        'token_level_rewards', 'route_gate', 'route_token_mask',
        'selective_route_mask', 'cfi_token_mask',
    )
    arrays = {k: _array(tensors[k]) for k in tensor_keys if k in tensors}
    for key in ('old_log_probs', 'teacher_log_probs', 'advantages', 'route_gate',
                'route_token_mask', 'response_mask', 'cfi_token_mask'):
        if key not in arrays:
            raise ValueError('Missing historical analysis signal: ' + key)
    arrays.update(
        graph_advantage=_array(advantage),
        raw_command_gap=_array(command_gap),
        student_entropy=_array(entropy),
        actor_eligible_mask=_array(actor_eligible_mask),
        raw_teacher_token_gap=_array(tensors['teacher_log_probs'] - tensors['old_log_probs']),
    )
    if observation_embeddings is not None:
        arrays['observation_embeddings'] = _array(observation_embeddings)

    owners = {}
    first_indices, inverse = [], []
    for index, row in enumerate(rows):
        key = (row['group'], row['trajectory'], row['turn'])
        if key not in owners:
            owners[key] = len(first_indices)
            first_indices.append(index)
        inverse.append(owners[key])
    arrays['unique_turn_indices'] = np.asarray(first_indices, dtype=np.int64)
    arrays['row_to_unique_turn'] = np.asarray(inverse, dtype=np.int64)
    np.savez_compressed(target / 'signals.npz', **arrays)
    del arrays

    inputs = tokenizer.batch_decode(tensors['prompts'], skip_special_tokens=True)
    outputs = tokenizer.batch_decode(tensors['responses'], skip_special_tokens=True)
    raw_obs = batch.non_tensor_batch['anchor_obs']
    with (target / 'rollout.jsonl').open('w', encoding='utf-8') as stream:
        for index, row in enumerate(rows):
            record = dict(row, row_index=index, unique_turn_index=inverse[index],
                          raw_observation=str(raw_obs[index]),
                          student_prompt=inputs[index], student_response=outputs[index])
            stream.write(json.dumps(_jsonable(record), ensure_ascii=False, allow_nan=False) + '\n')

    rs_arrays = {}
    for name, data in (('reflection', reflect_out), ('skill', prior_out),
                       ('rs_train', rs_train_batch)):
        if data is not None:
            for key in ('input_ids', 'responses', 'attention_mask', 'position_ids',
                        'old_log_probs', 'advantages'):
                if key in data.batch:
                    rs_arrays[f'{name}_{key}'] = _array(data.batch[key])
    np.savez_compressed(target / 'rs_tensors.npz', **rs_arrays)
    del rs_arrays
    reflections = (tokenizer.batch_decode(reflect_out.batch['responses'], skip_special_tokens=True)
                   if reflect_out is not None else [])
    if reflect_out is not None and len(reflections) != len(reflect_context['traj_list']):
        raise ValueError('Reflection text / trajectory order mismatch')
    rs_records = [dict(trajectory, reflection=reflections[index])
                  for index, trajectory in enumerate(reflect_context['traj_list'])] if reflections else []
    write_json(target / 'reflection_skill.json', {
        'reflections': rs_records,
        'uid_order': reflect_context['uid_order'],
        'prior_uids': reflect_context['prior_uids'],
        'skills': reflect_context['uid_to_prior'],
        'feedback': reflect_context['selective_feedback'],
    })
    write_json(target / 'config.json', config)
    summary = dict(
        step=int(step), natural_rollout=True, capture_stage='before_actor_and_rs_updates',
        padded_rows=count, unique_turns=len(first_indices),
        task_groups=len({r['group'] for r in rows}),
        trajectories=len({(r['group'], r['trajectory']) for r in rows}),
        reflection_count=len(reflections), skill_count=len(reflect_context['uid_to_prior']),
        reflection_order='reflect_context.traj_list', skill_order='reflect_context.prior_uids',
        rs_train_order='active reflections then active skills; use feedback and response presence',
        route_gate_includes_external_beta=False,
        coefficient_analysis_not_parameter_gradient=True,
        raw_gap_scope='all recorded response tokens; use masks for actual loss eligibility',
        model_weights_saved=False, full_vocabulary_logits_saved=False,
        tensor_rows_match_rollout_jsonl=True,
        files=[p.name for p in sorted(target.iterdir())],
    )
    write_json(target / 'summary.json', summary)
    size = sum(p.stat().st_size for p in target.iterdir() if p.is_file())
    return str(target), {'analysis/saved': 1., 'analysis/bytes': float(size),
                         'analysis/dump_seconds': time.monotonic() - started,
                         'analysis/unique_turns': float(len(first_indices))}
