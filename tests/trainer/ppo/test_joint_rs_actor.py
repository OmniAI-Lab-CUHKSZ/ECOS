"""Exercise the real Actor/R-S loss methods, not a reimplementation of PPO."""
import types
import os
import unittest
import sys
from pathlib import Path
import datetime
import json
import tempfile
import contextlib
import io
from collections import Counter
import torch
from omegaconf import OmegaConf
from verl.trainer.ppo.elastic_dispatch import pack_loss_packets, pack_joint_rs_packets
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_elastic_actor_execution import Toy, fixture, load_actor_methods


def distributed_worker(rank, world, store, result):
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(store).as_uri(), rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=90))
    try:
        actor = make_actor(diag=True)
        actor.config.rs_joint_diagnostics_first_last = True
        raw = actor.actor_module
        # Production no-LoRA uses flattened FSDP parameters, not orig_params.
        actor.actor_module = FSDP(raw, use_orig_params=False, device_id=torch.device('cpu'))
        actor.actor_optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=.01)
        def forward(micro_batch, **kwargs): return actor.actor_module(micro_batch['input_ids'], True)
        actor._forward_micro_batch = forward
        steps = []
        real_step = load_actor_methods(torch.device('cpu'))['_optimizer_step']
        def step(optimizer=None):
            steps.append(1)
            return real_step(actor, optimizer)
        actor._optimizer_step = step
        data = pack_loss_packets(fixture(16), logical_world_size=2, physical_world_size=world,
                                 global_minibatch_size=8, max_token_len=128)[rank]
        rs = pack_joint_rs_packets(rs_fixture(), physical_world_size=world,
                                   optimizer_steps=len(data.meta_info['elastic_loss_steps']), max_token_len=128)[rank]
        rs.meta_info['reflect_loss_coef'] = .1
        metrics = actor.update_policy(data, rs_data=rs)
        with FSDP.summon_full_params(actor.actor_module, writeback=False):
            output = dict(parameters=torch.cat([p.flatten() for p in raw.parameters()]).tolist(),
                          steps=len(steps), metrics=metrics)
        if rank == 0: Path(result).write_text(json.dumps(output))
    finally: dist.destroy_process_group()


def make_actor(lr=0.01, epochs=1, diag=False):
    raw = Toy()
    actor = types.SimpleNamespace(actor_module=raw, ulysses_sequence_parallel_size=1, reflect_optimizer=None)
    actor.config = OmegaConf.create(dict(use_dynamic_bsz=True, use_kl_loss=False,
        entropy_coeff=0., grad_clip=1., ppo_mini_batch_size=8, ppo_micro_batch_size_per_gpu=2,
        ppo_max_token_len_per_gpu=256, ppo_epochs=epochs, clip_ratio=.2,
        clip_ratio_low=.2, clip_ratio_high=.2, clip_ratio_c=3., loss_agg_mode='token-mean',
        policy_loss=dict(loss_mode='vanilla'), rs_joint_update=True, rs_joint_diagnostics=diag))
    actor.actor_optimizer = torch.optim.SGD(raw.parameters(), lr=lr)
    methods = load_actor_methods(torch.device('cpu'))
    for name in ('update_policy', 'update_reflect_policy'):
        setattr(actor, name, types.MethodType(methods[name], actor))
    actor.seen = []; actor.gradients = []
    def forward(micro_batch, **kwargs):
        actor.seen.extend(micro_batch['input_ids'][:, 0].tolist())
        return raw(micro_batch['input_ids'], True)
    actor._forward_micro_batch = forward
    def step(optimizer=None):
        actor.gradients.append([p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in raw.parameters()])
        norm = torch.sqrt(sum(g.square().sum() for g in actor.gradients[-1]))
        (optimizer or actor.actor_optimizer).step()
        return norm
    actor._optimizer_step = step
    return actor


def rs_fixture(n=17):
    data = fixture(n)
    data.batch['input_ids'] = data.batch['input_ids'] + 1000
    data.batch['row_index'] = torch.arange(n)
    # A mix of positive, zero, and negative group credits, no restandardization.
    data.batch['advantages'] = torch.linspace(-1., 1., n)[:, None] * data.batch['attention_mask']
    return data


class JointRSTest(unittest.TestCase):
    def test_actual_actor_action_teacher_outer_coefficient_002(self):
        from verl.trainer.ppo.selective_gigpo import token_route
        gradients=[]
        for coefficient in (.01,.02):
            actor=make_actor(lr=0.)
            main=fixture(16)
            response=main.batch['attention_mask'].bool()
            action=response & (torch.arange(response.shape[1])[None,:]%3==1)
            with torch.no_grad():
                _,old=actor.actor_module(main.batch['input_ids'],True)
            teacher=old+.1
            weights,selected,_=token_route(teacher,old,action,response,[.5]*len(main),[True]*len(main),
                1.,.5,actor_signal_scope='action_command',modulation_mode='symmetric_alignment')
            self.assertFalse(bool(weights[~action].any()))
            main.batch['old_log_probs']=old
            main.batch['teacher_log_probs']=teacher
            main.batch['advantages'].zero_()
            main.batch['route_gate']=weights
            main.batch['selective_route_mask']=selected
            main.meta_info['route_loss_coef']=coefficient
            packet=pack_loss_packets(main,logical_world_size=8,physical_world_size=1,
                                    global_minibatch_size=256,max_token_len=512)[0]
            actor.update_policy(packet)
            self.assertEqual(len(actor.gradients),1)
            gradients.append(torch.cat([g.flatten() for g in actor.gradients[0]]))
        self.assertGreater(float(gradients[1].norm()),0.)
        torch.testing.assert_close(gradients[1],2*gradients[0],rtol=1e-5,atol=1e-8)

    def test_rs_coefficient_scales_actual_gradient_tenfold(self):
        gradients = []
        for coefficient in (.01, .1):
            actor = make_actor(lr=0.)
            rs = pack_joint_rs_packets(rs_fixture(), physical_world_size=1,
                                       optimizer_steps=1, max_token_len=256)[0]
            rs.meta_info['reflect_loss_coef'] = coefficient
            actor.update_reflect_policy(rs, accumulate_only=True)
            gradients.append(torch.cat([p.grad.flatten() for p in actor.actor_module.parameters()]))
            self.assertEqual(len(actor.gradients), 0)
        self.assertGreater(gradients[0].norm().item(), 0.)
        torch.testing.assert_close(gradients[1], 10. * gradients[0], rtol=1e-5, atol=1e-8)

    def test_sparse_diagnostics_and_progress_leave_updates_unchanged(self):
        outputs = []
        for diag, first_last, progress in ((True, False, False), (True, True, True), (False, True, True)):
            main = pack_loss_packets(fixture(32), logical_world_size=2, physical_world_size=1,
                                     global_minibatch_size=8, max_token_len=256)[0]
            actor = make_actor(diag=diag)
            actor.config.rs_joint_diagnostics_first_last = first_last
            actor.config.rs_joint_progress = progress
            steps = len(main.meta_info['elastic_loss_steps'])
            self.assertGreater(steps, 2)
            rs = pack_joint_rs_packets(rs_fixture(), physical_world_size=1,
                                       optimizer_steps=steps, max_token_len=256)[0]
            rs.meta_info['reflect_loss_coef'] = .1
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                metrics = actor.update_policy(main, rs_data=rs)
            samples = (2 if first_last else steps) if diag else 0
            self.assertEqual(metrics['rs_joint/diagnostic_sample_count'], samples)
            self.assertEqual(len(metrics.get('rs_joint/main_grad_norm', [])), samples)
            self.assertEqual(len(metrics.get('rs_joint/parameter_update_norm', [])), samples)
            self.assertEqual(metrics['actor/optimizer_steps_attempted'], steps)
            self.assertEqual(metrics['actor/optimizer_steps_applied'], steps)
            for key in ('actor/grad_norm', 'rs_joint/auxiliary_backward_wall_s',
                        'rs_joint/main_backward_wall_s', 'rs_joint/diagnostics_and_step_wall_s'):
                self.assertEqual(len(metrics[key]), steps)
            if progress:
                for index in range(1, steps + 1):
                    self.assertIn(f'optimizer={index}/{steps} stage=main_backward', captured.getvalue())
                    self.assertIn(f'optimizer={index}/{steps} stage=complete applied=1', captured.getvalue())
            else:
                self.assertNotIn('[JointUpdate]', captured.getvalue())
            outputs.append((torch.cat([p.detach().flatten() for p in actor.actor_module.parameters()]), actor.seen))
        for parameters, seen in outputs[1:]:
            torch.testing.assert_close(parameters, outputs[0][0], rtol=0., atol=0.)
            self.assertEqual(seen, outputs[0][1])

    def test_single_slot_diagnostics_sample_only_once(self):
        actor = make_actor(diag=True)
        actor.config.rs_joint_diagnostics_first_last = True
        main = pack_loss_packets(fixture(8), logical_world_size=2, physical_world_size=1,
                                 global_minibatch_size=16, max_token_len=256)[0]
        rs = pack_joint_rs_packets(rs_fixture(), physical_world_size=1,
                                   optimizer_steps=1, max_token_len=256)[0]
        metrics = actor.update_policy(main, rs_data=rs)
        self.assertEqual(metrics['rs_joint/diagnostic_sample_count'], 1)
        self.assertEqual(len(metrics['rs_joint/parameter_update_norm']), 1)

    def test_nonfinite_gradient_skips_actual_optimizer(self):
        import torch.distributed as dist
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        with tempfile.TemporaryDirectory(prefix='joint_rs_nonfinite_') as directory:
            dist.init_process_group('gloo', init_method=(Path(directory)/'store').as_uri(), rank=0, world_size=1)
            try:
                actor = make_actor(diag=False)
                actor.actor_module = FSDP(actor.actor_module, use_orig_params=False, device_id=torch.device('cpu'))
                actor.actor_optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=.01)
                actor._forward_micro_batch = lambda micro_batch, **kwargs: actor.actor_module(micro_batch['input_ids'], True)
                actual_step = load_actor_methods(torch.device('cpu'))['_optimizer_step']
                actor._optimizer_step = types.MethodType(actual_step, actor)
                main = pack_loss_packets(fixture(8), logical_world_size=2, physical_world_size=1,
                                         global_minibatch_size=16, max_token_len=256)[0]
                main.batch['advantages'].fill_(float('nan'))
                before = [p.detach().clone() for p in actor.actor_module.parameters()]
                metrics = actor.update_policy(main)
                self.assertEqual(metrics['actor/optimizer_steps_attempted'], 1)
                self.assertEqual(metrics['actor/optimizer_steps_applied'], 0)
                for p, old in zip(actor.actor_module.parameters(), before):
                    torch.testing.assert_close(p, old, rtol=0., atol=0.)
            finally:
                dist.destroy_process_group()

    @unittest.skipUnless(os.name=='posix','CPU FSDP collective test requires Linux')
    def test_actual_fsdp_two_and_three_ranks(self):
        import torch.multiprocessing as mp
        results = []
        with tempfile.TemporaryDirectory(prefix='joint_rs_fsdp_') as directory:
            for world in (2, 3):
                path = Path(directory) / str(world); path.mkdir()
                mp.spawn(distributed_worker, args=(world, str(path/'store'), str(path/'out.json')), nprocs=world, join=True)
                results.append(json.loads((path/'out.json').read_text()))
        self.assertEqual(results[0]['steps'], 2)
        self.assertEqual(results[0]['steps'], results[1]['steps'])
        torch.testing.assert_close(torch.tensor(results[0]['parameters']), torch.tensor(results[1]['parameters']), rtol=1e-6, atol=1e-6)
        for key in ('rs_joint/main_grad_norm', 'rs_joint/scaled_rs_grad_norm', 'rs_joint/grad_cosine'):
            torch.testing.assert_close(torch.tensor(results[0]['metrics'][key]), torch.tensor(results[1]['metrics'][key]), rtol=1e-5, atol=1e-6)

    def test_state_center_and_fixed_stage_rewards(self):
        import numpy as np
        from verl.trainer.ppo.rs_sft_protocol import centered_credit, stage_weight
        rows = [dict(group='g', trajectory=str(i), turn=0) for i in range(8)]
        a = np.arange(8, dtype=float); gap = a - 3.5
        result = centered_credit(rows, a, gap, np.ones(8, bool), ['same']*8, limit=3.)
        shifted = centered_credit(rows, a+30., gap, np.ones(8, bool), ['same']*8, limit=3.)
        self.assertEqual(result['skill_reward'], shifted['skill_reward'])
        self.assertEqual(result['reflection_reward'], [3.]*8)
        self.assertEqual(result['skill_reward'], [3.])
        self.assertEqual(stage_weight('reflection', 16)*8, stage_weight('skill', 16))
        self.assertAlmostEqual(stage_weight('reflection',16)*128 + stage_weight('skill',16)*16, 1.)

    def test_no_row_replay_and_fixed_denominator_all_topologies(self):
        data = rs_fixture()
        for world in (1, 2, 3, 5, 8):
            for steps in (1, 2, 7, 20):
                weights = Counter(); counts = Counter()
                packets = pack_joint_rs_packets(data, physical_world_size=world, optimizer_steps=steps, max_token_len=64)
                for step in range(steps):
                    self.assertEqual(len({len(p.meta_info['elastic_loss_steps'][step]) for p in packets}), 1)
                for packet in packets:
                    for descriptors in packet.meta_info['elastic_loss_steps']:
                        for start, stop, scale, padding in descriptors:
                            if padding:
                                self.assertEqual(scale, 0.)
                                continue
                            for row in packet.batch['row_index'][start:stop].tolist():
                                weights[row] += scale / ((stop - start) * world)
                                counts[row] += 1
                self.assertEqual(counts, Counter({i: 1 for i in range(len(data))}))
                for weight in weights.values(): self.assertAlmostEqual(weight, 1 / len(data))

    def test_joint_gradient_equals_main_plus_scaled_rs_one_step(self):
        main = pack_loss_packets(fixture(8), logical_world_size=2, physical_world_size=1,
                                global_minibatch_size=16, max_token_len=256)[0]
        rs = pack_joint_rs_packets(rs_fixture(), physical_world_size=1, optimizer_steps=1, max_token_len=256)[0]
        joint = make_actor(diag=True)
        before = [p.clone() for p in joint.actor_module.parameters()]
        metrics = joint.update_policy(main, rs_data=rs)
        separate = make_actor(lr=0.)
        separate.update_policy(main)
        g_main = separate.gradients[0]
        separate.actor_optimizer.zero_grad()
        separate.update_reflect_policy(rs, accumulate_only=True)
        for expected, g, p, initial, new in zip(g_main, joint.gradients[0], separate.actor_module.parameters(), before, joint.actor_module.parameters()):
            torch.testing.assert_close(g, expected + p.grad)
            torch.testing.assert_close(new, initial - .01 * g)
        self.assertEqual(len(joint.gradients), 1)
        self.assertGreater(metrics['rs_joint/scaled_rs_grad_norm'][0], 0.)
        self.assertGreater(metrics['rs_joint/parameter_changed_fraction'][0], 0.)

    def test_ppo_epochs_do_not_replay_rs_or_add_optimizer_steps(self):
        main = pack_loss_packets(fixture(16), logical_world_size=2, physical_world_size=1,
                                global_minibatch_size=8, max_token_len=256)[0]
        actor = make_actor(epochs=2)
        rs = pack_joint_rs_packets(rs_fixture(), physical_world_size=1,
                                  optimizer_steps=len(main.meta_info['elastic_loss_steps']) * 2, max_token_len=256)[0]
        actor.update_policy(main, rs_data=rs)
        self.assertEqual(len(actor.gradients), len(main.meta_info['elastic_loss_steps']) * 2)
        actual = Counter(x for x in actor.seen if x >= 1000)
        expected = Counter(rs_fixture().batch['input_ids'][:, 0].tolist())
        self.assertEqual(actual, expected)

    def test_rs_credit_is_detached_and_negative_is_not_sft(self):
        gradients = []
        for sign in (1., -1., 0.):
            actor = make_actor()
            rs = rs_fixture(2)
            rs.batch['advantages'] = (torch.ones_like(rs.batch['old_log_probs']) * sign).requires_grad_()
            with torch.no_grad():
                _, old = actor.actor_module(rs.batch['input_ids'], True)
            rs.batch['old_log_probs'] = old.requires_grad_()
            rs = pack_joint_rs_packets(rs, physical_world_size=1, optimizer_steps=1, max_token_len=256)[0]
            rs.batch['advantages'].retain_grad(); rs.batch['old_log_probs'].retain_grad()
            actor.update_reflect_policy(rs, accumulate_only=True)
            self.assertIsNone(rs.batch['advantages'].grad)
            self.assertIsNone(rs.batch['old_log_probs'].grad)
            self.assertEqual(len(actor.gradients), 0)
            gradients.append(torch.cat([p.grad for p in actor.actor_module.parameters()]))
        torch.testing.assert_close(gradients[0], -gradients[1])
        torch.testing.assert_close(gradients[2], torch.zeros_like(gradients[2]))

    def test_empty_rs_does_not_change_legacy_main(self):
        data = pack_loss_packets(fixture(8), logical_world_size=2, physical_world_size=1,
                                 global_minibatch_size=16, max_token_len=256)[0]
        actor = make_actor()
        metrics = actor.update_policy(data)
        self.assertEqual(len(actor.gradients), 1)
        self.assertNotIn('rs/joint_update', metrics)


if __name__ == '__main__': unittest.main(verbosity=2)
