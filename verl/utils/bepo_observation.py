"""Observation-only features from the existing, frozen Qwen reference model.

No second model, adapter, future action, Teacher prompt, or reward is used.
The temporary norm hook reuses the existing command scorer's FSDP-safe path.
"""
import torch


def observation_inputs(tokenizer, observations, max_length=2048):
    # Match the offline plain-text encoding, with no chat template/history.
    unique=list(dict.fromkeys(str(x) for x in observations))
    encoded=[tokenizer.encode(s,add_special_tokens=True)[:max_length]
             or [tokenizer.eos_token_id] for s in unique]
    order=sorted(range(len(unique)),key=lambda i:len(encoded[i]))
    unique=[unique[i] for i in order];encoded=[encoded[i] for i in order]
    index={s:i for i,s in enumerate(unique)}
    width=max(map(len,encoded))
    ids=torch.full((len(encoded),width),tokenizer.pad_token_id,dtype=torch.long)
    mask=torch.zeros_like(ids);positions=torch.zeros_like(ids)
    for i,seq in enumerate(encoded):
        n=len(seq);ids[i,:n]=torch.tensor(seq);mask[i,:n]=1;positions[i,:n]=torch.arange(n)
    return dict(input_ids=ids,attention_mask=mask,position_ids=positions),[index[str(s)] for s in observations]


@torch.no_grad()
def reference_observation_features(module, tensors, forward, micro_batch_size=4):
    """Fixed microbatch count across FSDP ranks; right padding, last REAL token.

    The caller chooses the inference microbatch size. logits_to_keep
    avoids full vocabulary projection; the norm hook avoids retaining all layers.
    """
    if type(micro_batch_size) is not int or micro_batch_size < 1:
        raise ValueError('Observation microbatch size must be a positive integer')
    module.eval();features=[]
    for start in range(0,len(tensors['input_ids']),micro_batch_size):
        part={k:v[start:start+micro_batch_size] for k,v in tensors.items()}
        lengths=part['attention_mask'].sum(-1).long()
        if (lengths<=0).any():raise ValueError('Empty observation token sequence')
        width=int(lengths.max());part={k:v[:,:width] for k,v in part.items()}
        output,hidden=forward(module,need_hidden=True,final_hidden_only=True,
                              **part,use_cache=False,logits_to_keep=1)
        last=hidden[torch.arange(len(lengths),device=hidden.device),lengths-1].float()
        if not torch.isfinite(last).all() or (last.norm(dim=-1)==0).any():
            raise ValueError('Invalid frozen reference observation features')
        features.append(torch.nn.functional.normalize(last,dim=-1).cpu())
        del output,hidden
    return torch.cat(features)
