"""Build Teacher inputs from online Skills and exact Actor history token IDs."""

import torch
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask


def build_teacher_batch_with_text(
    batch: DataProto,
    row_skill_text,
    tokenizer,
    max_prompt_length: int,
    truncation: str = "left",
):
    """Build a teacher batch by prepending a PER-ROW privileged prior z_x.

    Preserve the original Actor history IDs and prepend the online group Skill.
    Rows with empty Skill text retain the ordinary Actor context.
    """
    from verl.utils.model import compute_position_id_with_mask

    bs = batch.batch["input_ids"].size(0)
    response_length = batch.batch["responses"].size(1)

    teacher_input_ids_list = []
    teacher_attention_mask_list = []
    teacher_position_ids_list = []
    actual_lens = []

    for i in range(bs):
        original_input_ids = batch.batch["input_ids"][i]
        original_attention_mask = batch.batch["attention_mask"][i]
        prompt_length = original_input_ids.size(0) - response_length

        prompt_ids = original_input_ids[:prompt_length]
        prompt_mask = original_attention_mask[:prompt_length]

        valid_start = prompt_mask.nonzero(as_tuple=True)[0]
        valid_start = valid_start[0].item() if len(valid_start) > 0 else 0
        valid_prompt_ids = prompt_ids[valid_start:]
        prompt_text = tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)

        skill_text = row_skill_text[i] if i < len(row_skill_text) else ""
        if skill_text:
            teacher_prompt_text = f"[Privileged Skill Information]\n{skill_text}\n\n" + prompt_text
        else:
            teacher_prompt_text = prompt_text

        # Preserve the actual Student history IDs; do not decode/re-tokenize it.
        prefix=tokenizer.encode(f"[Privileged Skill Information]\n{skill_text}\n\n",add_special_tokens=False) if skill_text else []
        teacher_prompt_ids=prefix+valid_prompt_ids.tolist()
        if len(teacher_prompt_ids) > max_prompt_length:
            if truncation=='error':raise ValueError('Skill plus exact Student prompt exceeds Teacher budget')
            teacher_prompt_ids = teacher_prompt_ids[-max_prompt_length:]

        teacher_prompt_ids = torch.tensor(teacher_prompt_ids, dtype=torch.long)
        actual_prompt_len = len(teacher_prompt_ids)

        response_ids = batch.batch["responses"][i]
        response_mask = original_attention_mask[-response_length:]

        teacher_full_ids = torch.cat([teacher_prompt_ids, response_ids])
        teacher_full_mask = torch.cat([
            torch.ones(actual_prompt_len, dtype=torch.long),
            response_mask,
        ])

        teacher_input_ids_list.append(teacher_full_ids)
        teacher_attention_mask_list.append(teacher_full_mask)
        actual_lens.append(actual_prompt_len)

    # Dynamic left-padding to the batch's longest actual prompt instead of the
    # fixed max_prompt_length: padding positions are masked anyway, and the
    # shorter sequence cuts the teacher forward cost (and its full-sequence
    # logits memory) several-fold.
    batch_max_len = max(actual_lens)
    for j in range(bs):
        pad = batch_max_len - actual_lens[j]
        if pad > 0:
            teacher_input_ids_list[j] = torch.cat([
                torch.full((pad,), tokenizer.pad_token_id, dtype=torch.long),
                teacher_input_ids_list[j],
            ])
            teacher_attention_mask_list[j] = torch.cat([
                torch.zeros(pad, dtype=torch.long),
                teacher_attention_mask_list[j],
            ])
        teacher_position_ids_list.append(
            compute_position_id_with_mask(teacher_attention_mask_list[j].unsqueeze(0))[0]
        )

    teacher_batch = DataProto.from_dict(
        tensors={
            "input_ids": torch.stack(teacher_input_ids_list),
            "attention_mask": torch.stack(teacher_attention_mask_list),
            "position_ids": torch.stack(teacher_position_ids_list),
            "responses": batch.batch["responses"],
        },
    )
    return teacher_batch
