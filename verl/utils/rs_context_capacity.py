"""Keep ordinary microbatches unchanged; accommodate a complete long sequence."""

def full_evidence_token_budget(attention_mask, ordinary_budget, model_capacity):
    lengths = attention_mask.sum(-1)
    longest = int(lengths.max().item())
    if int(lengths.min().item()) <= 0 or longest > int(model_capacity):
        raise ValueError(f'R/S sequence has {longest} tokens, outside model capacity {model_capacity}; no truncation')
    padded_width = int(attention_mask.shape[-1])
    if padded_width > int(model_capacity):
        raise ValueError(f'R/S padded width {padded_width} exceeds model capacity {model_capacity}; no truncation')
    return max(int(ordinary_budget), padded_width)
