"""Original ALFWorld response-only mean NLL positions, shared by CPU checks and FSDP."""
import torch


class ResponseLoss(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, prompt_ids, response_ids):
        device = next(self.model.parameters()).device
        ids = torch.tensor([prompt_ids + response_ids], device=device)
        hidden = self.model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                  use_cache=False).last_hidden_state
        hidden = hidden[0, len(prompt_ids)-1:len(prompt_ids)+len(response_ids)-1]
        logits = self.model.lm_head(hidden).float()
        labels = ids[0, len(prompt_ids):]
        logp = logits.gather(1, labels[:, None])[:, 0] - torch.logsumexp(logits, -1)
        return -logp.mean()
