"""Same joint-gradient diagnostics, with bounded device-side reductions.

Snapshots remain on CPU. CUDA reductions stream one 1M-element block at a
time instead of copying current tensors back and doing large CPU reductions.
Training tensors, gradient clipping, and optimizer operations are untouched.
"""
import math
import torch
import torch.distributed as dist

CHUNK_ELEMENTS = 1 << 20


def capture_gradients(module):
    return [None if p.grad is None else p.grad.detach().to(device='cpu', dtype=torch.float32, copy=True)
            for p in module.parameters()]


def capture_parameters(module):
    return [p.detach().to(device='cpu', dtype=torch.float32, copy=True) for p in module.parameters()]


def _totals(module, values):
    device = next(module.parameters()).device
    totals = torch.as_tensor(values, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(totals)
    return totals.cpu().tolist()


@torch.no_grad()
def gradient_components(module, before):
    device = next(module.parameters()).device
    if device.type == 'cuda':
        totals = torch.zeros(4, dtype=torch.float64, device=device)
        for p, old in zip(module.parameters(), before):
            if old is None and p.grad is None:
                continue
            current_flat = None if p.grad is None else p.grad.detach().reshape(-1)
            old_flat = None if old is None else old.reshape(-1)
            count = old.numel() if p.grad is None else p.grad.numel()
            for start in range(0, count, CHUNK_ELEMENTS):
                stop = min(start + CHUNK_ELEMENTS, count)
                current = (torch.zeros(stop-start, dtype=torch.float32, device=device)
                           if current_flat is None else current_flat[start:stop].float())
                main = (torch.zeros_like(current) if old_flat is None
                        else old_flat[start:stop].to(device=device, dtype=torch.float32))
                auxiliary = current - main
                totals += torch.stack((main.square().sum(dtype=torch.float64),
                    auxiliary.square().sum(dtype=torch.float64),
                    (main * auxiliary).sum(dtype=torch.float64),
                    current.square().sum(dtype=torch.float64)))
        main2, rs2, dot, joint2 = _totals(module, totals)
    else:
        main2 = rs2 = dot = joint2 = 0.
        for p, old in zip(module.parameters(), before):
            if old is None and p.grad is None:
                continue
            current = torch.zeros_like(old) if p.grad is None else p.grad.detach().float().cpu()
            main = torch.zeros_like(current) if old is None else old
            auxiliary = current - main
            main2 += main.square().sum(dtype=torch.float64).item()
            rs2 += auxiliary.square().sum(dtype=torch.float64).item()
            dot += (main * auxiliary).sum(dtype=torch.float64).item()
            joint2 += current.square().sum(dtype=torch.float64).item()
        main2, rs2, dot, joint2 = _totals(module, [main2, rs2, dot, joint2])
    main_norm, rs_norm = math.sqrt(main2), math.sqrt(rs2)
    return {'rs_joint/main_grad_norm': main_norm,
            'rs_joint/scaled_rs_grad_norm': rs_norm,
            'rs_joint/rs_to_main_grad_ratio': rs_norm / max(main_norm, 1e-30),
            'rs_joint/grad_cosine': dot / max(main_norm * rs_norm, 1e-30),
            'rs_joint/cosine_defined': float(main_norm > 0 and rs_norm > 0),
            'rs_joint/joint_grad_norm': math.sqrt(joint2)}


@torch.no_grad()
def parameter_change(module, before):
    device = next(module.parameters()).device
    fp32 = True
    if device.type == 'cuda':
        totals = torch.zeros(3, dtype=torch.float64, device=device)
        for p, old in zip(module.parameters(), before):
            current_flat, old_flat = p.detach().reshape(-1), old.reshape(-1)
            fp32 = fp32 and p.dtype == torch.float32
            for start in range(0, p.numel(), CHUNK_ELEMENTS):
                stop = min(start + CHUNK_ELEMENTS, p.numel())
                delta = current_flat[start:stop].float() - old_flat[start:stop].to(device=device)
                totals[0] += delta.square().sum(dtype=torch.float64)
                totals[1] += delta.numel()
                totals[2] += torch.count_nonzero(delta)
        square, count, nonzero = _totals(module, totals)
    else:
        square = count = nonzero = 0.
        for p, old in zip(module.parameters(), before):
            delta = p.detach().float().cpu() - old
            square += delta.square().sum(dtype=torch.float64).item()
            count += delta.numel()
            nonzero += torch.count_nonzero(delta).item()
            fp32 = fp32 and p.dtype == torch.float32
        square, count, nonzero = _totals(module, [square, count, nonzero])
    return {'rs_joint/parameter_update_norm': math.sqrt(square),
            'rs_joint/parameter_changed_fraction': nonzero / max(count, 1),
            'rs_joint/fp32_master_parameters': float(fp32)}
