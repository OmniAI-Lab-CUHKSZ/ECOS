"""Move one flat parameter's Adam state to the device only for its update.

The caller still clips the full reduced gradient and supplies the same AdamW
optimizer, gradients and groups. Each parameter is stepped exactly once using
the original PyTorch optimizer. Checkpoint keys/group structure stay unchanged.
"""
import torch


@torch.no_grad()
def streamed_adamw_step(optimizer):
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError('Streaming requires the original torch.optim.AdamW')
    groups=optimizer.param_groups
    if any(group.get(key,False) for group in groups for key in ['capturable','differentiable','fused']):
        raise ValueError('Capturable, differentiable or fused AdamW requires a separate implementation')
    try:
        for group in groups:
            for parameter in group['params']:
                if parameter.grad is None:continue
                state=optimizer.state[parameter]
                for key,value in list(state.items()):
                    if isinstance(value,torch.Tensor) and key!='step':
                        state[key]=value.to(parameter.device)
                optimizer.param_groups=[dict(group,params=[parameter])]
                optimizer.step()
                for key,value in list(optimizer.state[parameter].items()):
                    if isinstance(value,torch.Tensor):
                        optimizer.state[parameter][key]=value.to('cpu')
    finally:
        optimizer.param_groups=groups


def test(device='cpu'):
    import copy
    torch.manual_seed(987)
    reference=[torch.nn.Parameter(torch.randn(n,device=device)) for n in [41,89,7]]
    streamed=[torch.nn.Parameter(p.detach().clone()) for p in reference]
    def build(parameters):
        return torch.optim.AdamW([dict(params=parameters[:2],lr=1e-6),dict(params=parameters[2:],lr=2e-6)],betas=(.9,.999),weight_decay=.01,foreach=True)
    a=build(reference);b=build(streamed)
    for step in range(5):
        for index,(p,q) in enumerate(zip(reference,streamed)):
            grad=torch.randn_like(p);p.grad=grad;q.grad=grad.clone()
            if step==2 and index==1:p.grad=None;q.grad=None
        a.step();streamed_adamw_step(b)
        for p,q in zip(reference,streamed):
            assert torch.equal(p,q),(step,'parameter')
            for key,value in a.state[p].items():
                assert torch.equal(value.cpu(),b.state[q][key].cpu()),(step,key)
        assert a.state_dict()['param_groups']==b.state_dict()['param_groups']
        # Restore the existing Adam counters and moments midway through the test.
        if step==2:b.load_state_dict(copy.deepcopy(b.state_dict()))
    assert all(state['step'].item() in [4,5] for state in b.state.values())
    print('PASS exact AdamW parameters moments counters and groups with native-state reload on '+device,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');test(p.parse_args().device)
