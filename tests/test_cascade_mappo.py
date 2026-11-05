"""MAPPO objective, independent updates and the common cascade lifecycle."""
import copy
import csv
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
import yaml
from gym.spaces import Box, Discrete

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harl.algorithms.actors.happo import HAPPO
from harl.algorithms.actors.mappo import MAPPO
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.evaluation import evaluate_policy
from harl.envs.cascade_reservoir.periodic_evaluation import select_periodic_windows
from harl.runners.cascade_mappo_runner import CascadeMAPPORunner
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
from harl.runners.on_policy_ma_runner import OnPolicyMARunner


class MAPPOObjectiveTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.args = dict(hidden_sizes=[8, 8], activation_func='relu',
            use_feature_normalization=False, initialization_method='orthogonal_', gain=.01,
            lr=.003, opti_eps=1e-5, weight_decay=0., clip_param=.2, ppo_epoch=2,
            actor_num_mini_batch=1, entropy_coef=.01, use_max_grad_norm=True,
            max_grad_norm=.5, use_policy_active_masks=True, action_aggregation='prod',
            normalize_shared_advantages=True, gamma=.5, share_param=False,
            target_kl=None, kl_rollback=False, joint_target_kl=None)
        self.obs_space = Box(-np.inf, np.inf, (2,), dtype=np.float32)
        self.act_space = Discrete(3)

    def actor(self, cls=MAPPO, **changes):
        torch.manual_seed(21)
        return cls(dict(self.args, **changes), self.obs_space, self.act_space)

    def sample(self, actor):
        obs = np.array([[.2, -.3], [1., .5], [-.4, .8], [.9, -.7]], dtype=np.float32)
        actions = np.array([[0], [1], [2], [1]], dtype=np.float32)
        active = np.ones((4, 1), dtype=np.float32)
        with torch.no_grad():
            logp, _, _ = actor.evaluate_actions(obs, actions)
        old_logp = logp.numpy() + np.array([[.3], [-.3], [.2], [-.2]], dtype=np.float32)
        advantages = np.array([[-1.], [.2], [1.], [-.5]], dtype=np.float32)
        return (obs, actions, active.copy(), active, old_logp, advantages, None,
                np.ones_like(active), np.ones_like(active))

    def buffer(self, actor):
        sample = self.sample(actor)
        buffer = OnPolicyActorBuffer(dict(episode_length=4, n_rollout_threads=1),
                                     self.obs_space, self.act_space)
        buffer.obs[:-1, 0] = sample[0]
        buffer.actions[:, 0] = sample[1]
        with torch.no_grad():
            logp, _, _ = actor.evaluate_actions(sample[0], sample[1])
        buffer.action_log_probs[:, 0] = logp.numpy()
        buffer.masks[-1] = 0.
        buffer.update_factor(np.ones_like(buffer.active_masks[:-1]))
        return buffer, sample[5].reshape(4, 1, 1)

    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for x, y in zip(left, right):
                self.assert_state_equal(x, y)
        else:
            self.assertEqual(left, right)

    def test_reference_mappo_clipped_objective_and_gradient(self):
        actor = self.actor()
        reference = copy.deepcopy(actor)
        sample = self.sample(actor)
        # HARL's independent MAPPO objective on fully active rows and unit
        # calendar weights. Deliberately exercise both sides of PPO clipping.
        logp, entropy, _ = reference.evaluate_actions(
            sample[0], sample[1], active_masks=sample[3])
        ratio = torch.exp(logp - torch.from_numpy(sample[4])).prod(-1, keepdim=True)
        advantages = torch.from_numpy(sample[5])
        reference_loss = -torch.minimum(ratio*advantages,
            ratio.clamp(.8, 1.2)*advantages).sum(-1, keepdim=True).mean()
        reference.actor_optimizer.zero_grad()
        (reference_loss-reference.entropy_coef*entropy).backward()
        torch.nn.utils.clip_grad_norm_(reference.actor.parameters(), reference.max_grad_norm)
        reference.actor_optimizer.step()

        loss, actual_entropy, _, actual_ratio = actor.update(sample)
        torch.testing.assert_close(loss, reference_loss)
        torch.testing.assert_close(actual_entropy, entropy)
        torch.testing.assert_close(actual_ratio, ratio)
        for name, value in actor.actor.state_dict().items():
            torch.testing.assert_close(value, reference.actor.state_dict()[name], rtol=0., atol=1e-7)

    def test_unit_factor_matches_existing_happo_single_actor_update(self):
        actor, happo = self.actor(), self.actor(HAPPO)
        self.assert_state_equal(actor.actor.state_dict(), happo.actor.state_dict())
        sample = self.sample(actor)
        sample[-1][:] = np.array([[1.], [.5], [.25], [.125]], dtype=np.float32)
        left, right = actor.update(sample), happo.update(sample)
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, rtol=0., atol=0.)
        self.assert_state_equal(actor.actor.state_dict(), happo.actor.state_dict())

    def test_nonunit_or_missing_factors_are_rejected(self):
        actor = self.actor()
        for factor in (None, np.full((4, 1), .9), np.full((4, 1), np.nan),
                       np.ones((4, 2))):
            sample = list(self.sample(actor))
            sample[7] = factor
            with self.subTest(factor=factor), self.assertRaisesRegex(ValueError, 'unit buffer factors'):
                actor.update(tuple(sample))
        buffer, advantages = self.buffer(actor)
        buffer.factor[:] = 2.
        with self.assertRaisesRegex(ValueError, 'unit buffer factors'):
            actor.train(buffer, advantages, 'EP', diagnostics=True)

    def test_zero_control_or_zero_time_mass_does_not_advance_adam(self):
        actor = self.actor()
        actor.update(self.sample(actor))  # Populate Adam momentum first.
        model = copy.deepcopy(actor.actor.state_dict())
        optimizer = copy.deepcopy(actor.actor_optimizer.state_dict())
        for field in (3, 8):
            sample = list(self.sample(actor))
            sample[field] = np.zeros_like(sample[field])
            with mock.patch.object(actor, 'evaluate_actions',
                                   side_effect=AssertionError('zero-mass batch was evaluated')):
                result = actor.update(tuple(sample))
            self.assertEqual(len(result), 4)
            self.assertEqual(float(result[0]), 0.)
            self.assert_state_equal(model, actor.actor.state_dict())
            self.assert_state_equal(optimizer, actor.actor_optimizer.state_dict())

    def test_calendar_weights_and_diagnostics_preserve_update_and_rng(self):
        plain = self.actor()
        measured = copy.deepcopy(plain)
        buffer, advantages = self.buffer(plain)
        buffer.masks[2] = 0.  # An internal terminal resets the discount clock.
        buffer.active_masks[1] = 0.
        np.testing.assert_array_equal(buffer.discount_weights(.5).ravel(), [1., .5, 1., .5])
        measured_buffer = copy.deepcopy(buffer)
        torch.manual_seed(72)
        plain_info = plain.train(buffer, advantages.copy(), 'EP')
        rng_after_plain = torch.get_rng_state().clone()
        torch.manual_seed(72)
        info = measured.train(measured_buffer, advantages.copy(), 'EP', diagnostics=True)
        self.assertTrue(torch.equal(rng_after_plain, torch.get_rng_state()))
        self.assert_state_equal(plain.actor.state_dict(), measured.actor.state_dict())
        self.assert_state_equal(plain.actor_optimizer.state_dict(), measured.actor_optimizer.state_dict())
        self.assertEqual(info['optimizer_steps'], 2.)
        self.assertEqual(info['ratio_samples'], 6.)
        self.assertEqual(plain_info['ratio_samples'], info['ratio_samples'])
        for suffix in ('before', 'after'):
            self.assertAlmostEqual(info['regularized_objective_'+suffix],
                info['clipped_surrogate_'+suffix]+measured.entropy_coef*info['weighted_entropy_'+suffix])

    def test_local_kl_rollback_restores_rejected_candidate(self):
        actor = self.actor(lr=.5, target_kl=1e-14, kl_rollback=True)
        buffer, advantages = self.buffer(actor)
        model = copy.deepcopy(actor.actor.state_dict())
        optimizer = copy.deepcopy(actor.actor_optimizer.state_dict())
        info = actor.train(buffer, advantages, 'EP', diagnostics=True)
        self.assertEqual(info['optimizer_rejected_steps'], 1.)
        self.assertEqual(info['optimizer_steps'], 0.)
        self.assertEqual(info['kl_early_stopped'], 1.)
        self.assertLessEqual(info['policy_kl'], 1e-14)
        self.assert_state_equal(model, actor.actor.state_dict())
        self.assert_state_equal(optimizer, actor.actor_optimizer.state_dict())

    def test_shared_parameters_and_joint_sequential_budget_are_explicitly_unsupported(self):
        for changes in (dict(share_param=True),
                        dict(target_kl=.01, kl_rollback=True, joint_target_kl=.03)):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'MAPPO'):
                self.actor(**changes)


class MAPPORunnerDispatchTest(unittest.TestCase):
    def test_own_return_channels_without_prepost_likelihood_evaluation(self):
        calls, captured = [], {}
        runner = OnPolicyMARunner()
        runner.num_agents = 5
        runner.algo_args = dict(algo=dict(share_param=False, joint_target_kl=None,
                                        normalize_shared_advantages=True))
        runner.diagnostics_enabled = True
        raw = np.arange(24, dtype=np.float32).reshape(4, 1, 6)
        runner.critic_buffer = SimpleNamespace(returns=raw**2, value_preds=raw/7.)
        runner.value_normalizer = SimpleNamespace(denormalize=lambda values: 2.*values)

        class Buffer:
            active_masks = np.ones((4, 1, 1), dtype=np.float32)
            def update_factor(self, factor):
                self.factor = factor.copy()

        class Actor:
            def __init__(self, index):
                self.index = index
            def train(self, buffer, advantages, state_type, diagnostics=False):
                calls.append(self.index)
                captured[self.index] = advantages.copy()
                np.testing.assert_array_equal(buffer.factor, np.ones((3, 1, 1)))
                if state_type != 'EP' or not diagnostics:
                    raise AssertionError('Wrong training options')
                return dict(policy_loss=float(self.index))
            def evaluate_actions(self, *args, **kwargs):
                raise AssertionError('MAPPO runner must not evaluate cross-agent likelihood factors')

        class Critic:
            def train(self, buffer, normalizer):
                calls.append('critic')
                return dict(value_loss=0.)

        runner.actor_buffer = [Buffer() for _ in range(5)]
        runner.actor = [Actor(i) for i in range(5)]
        runner.critic = Critic()
        infos, critic_info = runner.train()
        expected = raw[:-1]**2-2.*raw[:-1]/7.
        expected = (expected-expected.mean(axis=(0, 1), keepdims=True))/(
            expected.std(axis=(0, 1), keepdims=True)+1e-5)
        self.assertEqual(calls, [0, 1, 2, 3, 4, 'critic'])
        for i in range(5):
            np.testing.assert_array_equal(captured[i], expected[..., i:i+1])
            self.assertEqual(infos[i]['factor_min'], 1.)
            self.assertEqual(infos[i]['factor_max'], 1.)
        self.assertIsNone(runner.joint_kl_info)
        self.assertEqual(critic_info['value_loss'], 0.)


class CascadeMAPPOIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = yaml.safe_load((ROOT/'harl/configs/algos_cfgs/happo_cascade.yaml').read_text())
        cls.physical = yaml.safe_load((ROOT/'harl/configs/envs_cfgs/cascade_reservoir.yaml').read_text())
        cls.args['train'].update(episode_length=4, n_rollout_threads=2)
        cls.args['model'].update(hidden_sizes=[8, 8], action_bins=7)
        cls.args['algo'].update(ppo_epoch=2, critic_epoch=2, target_kl=None,
                                kl_rollback=False, joint_target_kl=None)
        cls.args['experiment'].update(episodes=2, seeds=[1], diagnostics=True)
        cls.data = CascadeData(ROOT/'data/cascade_reservoir_new', cls.physical, cls.args['experiment'])
        cls.windows = select_periodic_windows(cls.data, episode_days=4)

    def test_same_seed_initializes_identical_independent_networks(self):
        happo = CascadeReservoirRunner(self.data, self.physical, self.args, 1,
                                        periodic_windows=self.windows)
        self.addCleanup(happo.close)
        happo_rng = torch.get_rng_state().clone()
        mappo = CascadeMAPPORunner(self.data, self.physical, self.args, 1,
                                   periodic_windows=self.windows)
        self.addCleanup(mappo.close)
        self.assertTrue(torch.equal(happo_rng, torch.get_rng_state()))
        self.assertEqual(len({id(a.actor) for a in mappo.actor}), 5)
        for ha, ma in zip(happo.actor, mappo.actor):
            self.assertIsInstance(ma, MAPPO)
            for name, value in ha.actor.state_dict().items():
                self.assertTrue(torch.equal(value, ma.actor.state_dict()[name]))
        for name, value in happo.critic.critic.state_dict().items():
            self.assertTrue(torch.equal(value, mappo.critic.critic.state_dict()[name]))
        self.assertEqual(mappo.critic_output_dim, 6)

    def test_tiny_real_update_diagnostics_checkpoint_and_replay(self):
        from harl.envs.cascade_reservoir.checkpoint import load_checkpoint
        from harl.envs.cascade_reservoir.trajectory_diagnostics import TrajectoryDiagnosticsWriter

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            directory = Path(directory)
            trace = directory/'trajectory.csv'
            with TrajectoryDiagnosticsWriter(trace) as writer:
                runner = CascadeMAPPORunner(self.data, self.physical, self.args, 1,
                    periodic_windows=self.windows, trajectory_log=writer,
                    checkpoint_dir=directory/'checkpoints')
                self.addCleanup(runner.close)
                initial_actor = copy.deepcopy(runner.actor[0].actor.state_dict())
                initial_critic = copy.deepcopy(runner.critic.critic.state_dict())
                with mock.patch('harl.runners.on_policy_ha_runner.OnPolicyHARunner.train',
                                side_effect=AssertionError('HAPPO dispatch used for MAPPO')):
                    runner.run(2)
            self.assertEqual(runner.update_count, 1)
            self.assertEqual(len(runner.diagnostics_history), 1)
            self.assertTrue(any(not torch.equal(value, runner.actor[0].actor.state_dict()[name])
                                for name, value in initial_actor.items()))
            self.assertTrue(any(not torch.equal(value, runner.critic.critic.state_dict()[name])
                                for name, value in initial_critic.items()))
            for i, buffer in enumerate(runner.actor_buffer):
                np.testing.assert_array_equal(buffer.factor, np.ones((4, 2, 1)))
                self.assertEqual(runner.training_history[0]['actor'+str(i)+'_factor_max'], 1.)
            with trace.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4*2*5)
            self.assertTrue(all(float(row['happo_factor']) == 1. for row in rows))
            checkpoint = directory/'checkpoints/update_000001.pt'
            rng = torch.get_rng_state().clone()
            policy = load_checkpoint(checkpoint)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            replay = policy.replay([1965])
            reference = evaluate_policy(runner.actor, self.data, self.physical,
                seed=1, episode=2, training_steps=8, gamma=self.args['algo']['gamma'],
                source_years=[1965], episode_days=4, diagnostics=True)
            for field in ('energy_MWh', 'episode_reward', 'episode_mixed_rewards',
                          'episode_system_reward', 'spill_volume_m3'):
                np.testing.assert_allclose(replay[field], reference[field], rtol=0., atol=1e-10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
