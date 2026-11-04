"""Exercise the real training, diagnostic, checkpoint and replay entry points."""
import copy
import csv
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import yaml
from gym.spaces import Box, Discrete

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.evaluation import evaluate_policy
from harl.envs.cascade_reservoir.periodic_evaluation import select_periodic_windows
from harl.envs.cascade_reservoir.reward import (
    EpisodeRewardMetrics, OperationReward, reward_metadata, reward_settings)
from harl.algorithms.actors.happo import (
    HAPPO, validate_guarantee_auxiliary_coef, validate_guarantee_auxiliary_mode)
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner


class GuaranteedOutputFrequencyRewardTest(unittest.TestCase):
    def setUp(self):
        self.data = SimpleNamespace(power_cap=np.full(5, 1000.),
            service_requirements=SimpleNamespace(
                guaranteed_output_mw=np.array([100., 100., 100., 100., 0.]),
                ecological_flow_m3s=np.full(5, 100.), metadata=lambda: {}))

    def transition(self, power, weight=None, enabled=True):
        penalties = dict(enabled=enabled, guaranteed_output_weight=1., ecological_flow_weight=1.)
        if weight is not None:
            penalties['guaranteed_output_violation_weight'] = weight
        settings = reward_settings(dict(reward_penalties=penalties))
        power = np.asarray(power, dtype=np.float64)
        result = dict(power=power.copy(), release=np.full(5, 100.),
            reward=float(power.sum()/self.data.power_cap.sum()),
            energy=24.*power, actions=np.arange(5),
            storage_start=np.arange(5, dtype=float), storage_end=np.arange(5, dtype=float)+1.)
        OperationReward(self.data, settings).apply(result)
        return result

    def test_frequency_weight_is_optional_and_validated(self):
        self.assertEqual(reward_settings({})['guaranteed_output_violation_weight'], 0.)
        default = self.transition([90., 100., 100., 100., 0.])
        zero = self.transition([90., 100., 100., 100., 0.], weight=0.)
        for field in ('local_rewards', 'mixed_rewards', 'local_reward_penalties',
                      'local_guaranteed_output_violation_penalties'):
            np.testing.assert_array_equal(default[field], zero[field])
        self.assertEqual(default['reward'], zero['reward'])
        self.assertAlmostEqual(default['reward'], .053)
        np.testing.assert_allclose(default['local_rewards'], [-.01, .1, .1, .1, 0.], atol=1e-15)
        for value in (-1., float('inf'), float('nan'), True, np.bool_(False), 'invalid'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
                reward_settings(dict(reward_penalties=dict(guaranteed_output_violation_weight=value)))
        self.assertEqual(reward_settings(dict(reward_penalties=dict(
            guaranteed_output_violation_weight=1.25)))['guaranteed_output_violation_weight'], 1.25)

    def test_local_cost_preserves_physics_legacy_reward_and_service_counts(self):
        power = [99.999, 50., 100., 150., 0.]
        baseline, changed = self.transition(power, 0.), self.transition(power, 2.)
        frequency = np.array([2., 2., 0., 0., 0.])
        np.testing.assert_array_equal(changed['local_guaranteed_output_violation_penalties'], frequency)
        np.testing.assert_allclose(changed['mixed_rewards'], baseline['mixed_rewards']-.5*frequency)
        self.assertEqual(changed['global_reward'], baseline['global_reward'])
        self.assertAlmostEqual(changed['system_reward'], baseline['system_reward']-.5*frequency.sum())
        for field in ('reward', 'generation_reward', 'reward_penalty', 'reward_penalty_components'):
            self.assertEqual(changed[field], baseline[field])
        for field in ('power', 'release', 'energy', 'actions', 'storage_start', 'storage_end',
                      'reward_penalty_by_reservoir'):
            np.testing.assert_array_equal(changed[field], baseline[field])
        for group in ('service_shortfalls', 'normalized_service_shortfalls', 'service_active_masks',
                      'service_shortfall_integrals'):
            for term in baseline[group]:
                np.testing.assert_array_equal(changed[group][term], baseline[group][term])
        before, after = EpisodeRewardMetrics(.5), EpisodeRewardMetrics(.5)
        for _ in range(2):
            before.update(baseline)
            after.update(changed)
        self.assertEqual(before.shortfall_summary(), after.shortfall_summary())
        self.assertEqual(before.summary()['episode_reward'], after.summary()['episode_reward'])
        learning = after.learning_summary()
        np.testing.assert_array_equal(learning['episode_local_guaranteed_output_violation_penalties'],
                                      1.5*frequency)
        np.testing.assert_allclose(learning['episode_local_reward_penalties'],
                                  np.sum(list(learning['episode_local_penalty_components'].values()), axis=0))
        metadata = reward_metadata(self.data)
        self.assertTrue(metadata['actor_reward_definition'].endswith('_v2'))
        self.assertTrue(metadata['system_reward_definition'].endswith('_v2'))
        self.assertEqual(metadata['reward_definition'], 'generation_minus_mean_service_shortfalls_v1')

    def test_frequency_indicator_uses_existing_tolerance_and_inactive_requirements(self):
        power = [100., 100.-2.5e-9, 100.-4.e-8, 101., 0.]
        penalized = self.transition(power, 2.)
        disabled = self.transition(power, 2., enabled=False)
        np.testing.assert_array_equal(penalized['local_guaranteed_output_violation_penalties'],
                                      [0., 0., 2., 0., 0.])
        np.testing.assert_array_equal(disabled['local_guaranteed_output_violation_penalties'], np.zeros(5))
        np.testing.assert_array_equal(disabled['local_reward_penalties'], np.zeros(5))
        enabled_metrics, disabled_metrics = EpisodeRewardMetrics(1.), EpisodeRewardMetrics(1.)
        enabled_metrics.update(penalized)
        disabled_metrics.update(disabled)
        self.assertEqual(enabled_metrics.shortfall_summary(), disabled_metrics.shortfall_summary())
        guarantee = enabled_metrics.shortfall_summary()['guaranteed_output']
        self.assertEqual(guarantee['reservoir_day_count'], 1)
        self.assertEqual(guarantee['active_reservoir_days'], 4)
        unavailable = SimpleNamespace(power_cap=self.data.power_cap, service_requirements=None)
        transition = dict(power=np.ones(5), reward=.001)
        OperationReward(unavailable, reward_settings(dict(reward_penalties=dict(
            enabled=False, guaranteed_output_violation_weight=2.)))).apply(transition)
        np.testing.assert_array_equal(transition['local_guaranteed_output_violation_penalties'], np.zeros(5))

    def test_equal_total_deficit_is_distinguished_by_violation_frequency(self):
        fewer, more = EpisodeRewardMetrics(1.), EpisodeRewardMetrics(1.)
        for metric, powers in ((fewer, (90., 100.)), (more, (95., 95.))):
            for power in powers:
                metric.update(self.transition([power, 100., 100., 100., 0.], 2.))
        self.assertAlmostEqual(fewer.summary()['episode_reward'], more.summary()['episode_reward'])
        self.assertEqual(fewer.shortfall_summary()['guaranteed_output']['reservoir_day_count'], 1)
        self.assertEqual(more.shortfall_summary()['guaranteed_output']['reservoir_day_count'], 2)
        self.assertAlmostEqual(fewer.learning_summary()['episode_mixed_rewards'][0]-
                               more.learning_summary()['episode_mixed_rewards'][0], 1.)


class GuaranteeAuxiliaryLearningTest(unittest.TestCase):
    @staticmethod
    def actor(coef=1., mode='expected_cost', water_weight=.1):
        config = yaml.safe_load((ROOT/'harl/configs/algos_cfgs/happo_cascade.yaml').read_text())
        args = {**config['model'], **config['algo']}
        args.update(hidden_sizes=[8], lr=.01, entropy_coef=0., ppo_epoch=2,
                    target_kl=None, kl_rollback=False, joint_target_kl=None,
                    guarantee_auxiliary_coef=coef, guarantee_auxiliary_mode=mode,
                    guarantee_auxiliary_water_weight=water_weight, gamma=1.)
        torch.set_num_threads(1)
        torch.manual_seed(17)
        actor = HAPPO(args, Box(-np.inf, np.inf, (2,), dtype=np.float32), Discrete(3))
        with torch.no_grad():
            actor.actor.act.action_out.linear.weight.zero_()
            actor.actor.act.action_out.linear.bias.zero_()
        return actor

    @staticmethod
    def sample(actor, costs=None, active=1., weight=1., advantage=0.):
        obs = np.zeros((1, 2), dtype=np.float32)
        actions = np.zeros((1, 1), dtype=np.float32)
        with torch.no_grad():
            logp = actor.evaluate_actions(obs, actions)[0].numpy()
        one = np.ones((1, 1), dtype=np.float32)
        sample = (obs, actions, one.copy(), one*active, logp, one*advantage,
                  np.ones((1, 3), dtype=np.float32), one.copy(), one*weight)
        return sample if costs is None else sample+(np.asarray(costs, dtype=np.float32),)

    @staticmethod
    def probabilities(actor):
        with torch.no_grad():
            return actor.evaluate_actions(np.zeros((1, 2), dtype=np.float32),
                                          np.zeros((1, 1), dtype=np.float32))[2].probs.clone()

    def test_dense_cost_moves_probability_without_sampled_ppo_advantage(self):
        actor = self.actor()
        before = self.probabilities(actor)
        sample = self.sample(actor, [[1., 0., 1.]])
        action = sample[1].copy()
        result = actor.update(sample)
        after = self.probabilities(actor)
        self.assertEqual(len(result), 5)
        self.assertAlmostEqual(result[0].item(), 0.)
        self.assertAlmostEqual(result[4].item(), 2./3., places=6)
        self.assertGreater(after[0, 1].item(), before[0, 1].item())
        self.assertLess((after[0, 0]+after[0, 2]).item(), 2./3.)
        np.testing.assert_array_equal(sample[1], action)

    def test_exact_expectation_masks_discount_factor_and_constant_cost_gradient(self):
        actor = self.actor()
        logits = torch.tensor([[.2, -.1, .3], [.1, .2, -.3]], requires_grad=True)
        factor = torch.tensor([[2.], [9.]], requires_grad=True)
        costs = torch.tensor([[1., 0., 1.], [0., 1., 0.]], requires_grad=True)
        distribution = torch.distributions.Categorical(logits=logits)
        loss = actor._guarantee_auxiliary_cost(distribution, costs, torch.tensor([[.5], [0.]]), factor)
        expected = .5*(distribution.probs[0, 0]+distribution.probs[0, 2])
        self.assertAlmostEqual(loss.item(), expected.item(), places=7)
        loss.backward()
        torch.testing.assert_close(logits.grad[1], torch.zeros(3), rtol=0., atol=0.)
        self.assertIsNone(factor.grad)
        self.assertIsNone(costs.grad)
        for constant in (0., 1., 2.75):
            with self.subTest(constant=constant):
                logits = torch.tensor([[.2, -.1, .3]], requires_grad=True)
                value = actor._guarantee_auxiliary_cost(torch.distributions.Categorical(logits=logits),
                    np.full((1, 3), constant, np.float32), torch.ones(1, 1), torch.ones(1, 1))
                value.backward()
                torch.testing.assert_close(logits.grad, torch.zeros_like(logits), rtol=0., atol=0.)

    def test_continuous_water_cost_has_exact_expected_cost_and_gradient(self):
        actor = self.actor()
        logits = torch.tensor([[.2, -.1, .3]], requires_grad=True)
        costs = torch.tensor([[1., .025, 1.1]])
        distribution = torch.distributions.Categorical(logits=logits)
        expected = (distribution.probs.detach()*costs).sum(dim=-1, keepdim=True)
        loss = actor._guarantee_auxiliary_cost(distribution, costs, torch.tensor([[.5]]),
                                             torch.tensor([[2.]]))
        torch.testing.assert_close(loss, expected.squeeze(), rtol=1e-6, atol=1e-7)
        loss.backward()
        torch.testing.assert_close(logits.grad, distribution.probs.detach()*(costs-expected),
                                   rtol=1e-6, atol=1e-7)

    def test_feasible_imitation_moves_teacher_probability_without_changing_action(self):
        actor = self.actor(mode='feasible_imitation')
        before = self.probabilities(actor)
        sample = self.sample(actor, [[1., .025, .1]])
        sampled_action = sample[1].copy()
        random_state = torch.get_rng_state().clone()
        result = actor.update(sample)
        self.assertTrue(torch.equal(random_state, torch.get_rng_state()))
        self.assertAlmostEqual(result[4].item(), np.log(3.), places=6)
        self.assertGreater(self.probabilities(actor)[0, 1].item(), before[0, 1].item())
        np.testing.assert_array_equal(sample[1], sampled_action)

    def test_feasible_imitation_matches_reference_and_skips_unreachable_constant_rows(self):
        actor = self.actor(mode='feasible_imitation')
        initial = [[.2, -.1, .3], [.1, .2, -.3], [-.2, .4, .1], [.3, -.4, .2]]
        logits = torch.tensor(initial, requires_grad=True)
        reference_logits = torch.tensor(initial, requires_grad=True)
        costs = torch.tensor([[1., .025, .1], [1., 1.025, 1.1], [.2, .2, .2], [.1, .05, 1.1]])
        weights, factor = torch.tensor([[.5], [1.], [1.], [0.]]), torch.tensor(
            [[2.], [9.], [3.], [4.]], requires_grad=True)
        loss = actor._guarantee_auxiliary_cost(torch.distributions.Categorical(logits=logits),
                                             costs, weights, factor)
        minimum, target = costs.min(dim=-1, keepdim=True)
        valid = (minimum < 1.) & (costs.max(dim=-1, keepdim=True).values > minimum)
        distribution = torch.distributions.Categorical(logits=reference_logits)
        reference = (weights*factor.detach()*valid*(-distribution.logits.gather(-1, target))).mean()
        torch.testing.assert_close(loss, reference, rtol=0., atol=0.)
        loss.backward()
        reference.backward()
        torch.testing.assert_close(logits.grad, reference_logits.grad, rtol=0., atol=0.)
        torch.testing.assert_close(logits.grad[1:], torch.zeros(3, 3), rtol=0., atol=0.)
        self.assertIsNone(factor.grad)
        for constant in (0., .2, 1., 2.):
            constant_logits = torch.tensor([[.2, -.1, .3]], requires_grad=True)
            value = actor._guarantee_auxiliary_cost(
                torch.distributions.Categorical(logits=constant_logits),
                np.full((1, 3), constant, np.float32), torch.ones(1, 1), torch.ones(1, 1))
            value.backward()
            self.assertEqual(value.item(), 0.)
            torch.testing.assert_close(constant_logits.grad, torch.zeros_like(constant_logits), rtol=0., atol=0.)

    def test_auxiliary_modes_and_teacher_water_range_are_validated(self):
        for mode in ('invalid', None, 1):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_guarantee_auxiliary_mode(mode, 1., .1)
        for weight in (-.1, float('nan'), float('inf'), 1., 2., .9999999999):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                validate_guarantee_auxiliary_mode('feasible_imitation', 1., weight)
        self.assertEqual(validate_guarantee_auxiliary_mode('feasible_imitation', 0., 2.),
                         'feasible_imitation')
        self.assertEqual(validate_guarantee_auxiliary_mode('expected_cost', 1., 2.), 'expected_cost')
        plain, teacher = self.actor(0.), self.actor(0., mode='feasible_imitation')
        plain.update(self.sample(plain, advantage=1.))
        teacher.update(self.sample(teacher, [[1., .025, .1]], advantage=1.))
        for name, value in plain.actor.state_dict().items():
            self.assertTrue(torch.equal(value, teacher.actor.state_dict()[name]))

    def test_forced_and_zero_weight_batches_do_not_move_adam(self):
        for mode in ('expected_cost', 'feasible_imitation'):
            with self.subTest(mode=mode):
                actor = self.actor(mode=mode)
                actor.update(self.sample(actor, [[1., 0., 1.]]))
                parameters = copy.deepcopy(actor.actor.state_dict())
                steps = [state['step'].clone() for state in actor.actor_optimizer.state.values()]
                for active, weight in ((0., 1.), (1., 0.)):
                    actor.update(self.sample(actor, [[0., 1., 0.]], active=active, weight=weight))
                    for name, value in parameters.items():
                        self.assertTrue(torch.equal(value, actor.actor.state_dict()[name]))
                    for before, state in zip(steps, actor.actor_optimizer.state.values()):
                        self.assertTrue(torch.equal(before, state['step']))

    def test_training_and_diagnostics_include_the_same_auxiliary_objective(self):
        for mode in ('expected_cost', 'feasible_imitation'):
            with self.subTest(mode=mode):
                actor = self.actor(mode=mode)
                buffer = OnPolicyActorBuffer(dict(episode_length=2, n_rollout_threads=1),
                    Box(-np.inf, np.inf, (2,), dtype=np.float32), Discrete(3))
                for t in range(2):
                    buffer.insert(np.zeros((1, 2), np.float32), np.zeros((1, 1), np.float32),
                        np.full((1, 1), -np.log(3.), np.float32), np.full((1, 1), float(t == 0), np.float32),
                        guarantee_action_costs=np.array([[1., 0., 1.]], np.float32))
                buffer.update_factor(np.ones((2, 1, 1), np.float32))
                info = actor.train(buffer, np.zeros((2, 1, 1), np.float32), 'EP', diagnostics=True)
                self.assertEqual(info['optimizer_steps'], 2.)
                self.assertLess(info['guarantee_auxiliary_cost_after'], info['guarantee_auxiliary_cost_before'])
                self.assertAlmostEqual(info['regularized_objective_after'], -info['guarantee_auxiliary_cost_after'])
                self.assertAlmostEqual(info['clipped_surrogate_after'], 0.)

    def test_zero_coefficient_keeps_original_update_and_rejects_invalid_costs_when_enabled(self):
        plain, labelled = self.actor(0.), self.actor(0.)
        first = plain.update(self.sample(plain, advantage=1.))
        second = labelled.update(self.sample(labelled, [[1., 0., 1.]], advantage=1.))
        self.assertEqual(len(first), 4)
        for name, value in plain.actor.state_dict().items():
            self.assertTrue(torch.equal(value, labelled.actor.state_dict()[name]))
        for left, right in zip(first, second):
            torch.testing.assert_close(left, right, rtol=0., atol=0.)
        for value in (-1., float('inf'), float('nan'), True, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_guarantee_auxiliary_coef(value)
        actor = self.actor()
        for costs in (None, [[0., 1.]], [[0., -.5, 1.]], [[0., float('nan'), 1.]]):
            with self.subTest(costs=costs), self.assertRaises(ValueError):
                actor.update(self.sample(actor, costs))

    def test_buffer_costs_stay_aligned_after_permutation_and_missing_rows_fail(self):
        args = dict(episode_length=3, n_rollout_threads=2)
        space = Box(-np.inf, np.inf, (2,), dtype=np.float32)
        buffer = OnPolicyActorBuffer(args, space, Discrete(3))
        buffer.obs[0, :, 0] = [0., 1.]
        for t in range(3):
            rows = np.arange(t*2, t*2+2)
            costs = (rows[:, None] % 3 == np.arange(3)[None, :]).astype(np.float32)
            obs = np.zeros((2, 2), np.float32)
            obs[:, 0] = rows+2
            buffer.insert(obs, np.zeros((2, 1)), np.zeros((2, 1)), np.ones((2, 1)),
                          guarantee_action_costs=costs)
        advantages = np.arange(6, dtype=np.float32).reshape(3, 2, 1)
        buffer.update_factor(advantages+1.)
        weights = buffer.discount_weights(.5)
        seen = []
        for sample in buffer.feed_forward_generator_actor(advantages, 2, weights):
            self.assertEqual(len(sample), 10)
            ids = sample[0][:, 0].astype(int)
            seen.extend(ids.tolist())
            np.testing.assert_array_equal(sample[9], ids[:, None] % 3 == np.arange(3)[None, :])
            np.testing.assert_array_equal(sample[5][:, 0], ids)
            np.testing.assert_array_equal(sample[7][:, 0], ids+1.)
            np.testing.assert_array_equal(sample[8][:, 0], .5**(ids//2))
        self.assertEqual(sorted(seen), list(range(6)))
        buffer.guarantee_action_costs[1] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite nonnegative'):
            list(buffer.feed_forward_generator_actor(advantages, 1, weights))
        plain = OnPolicyActorBuffer(args, space, Discrete(3))
        plain.update_factor(np.ones((3, 2, 1), np.float32))
        self.assertEqual(len(next(plain.feed_forward_generator_actor(advantages, 1, weights))), 9)


class MissingWorkflowIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = yaml.safe_load((ROOT / 'harl/configs/algos_cfgs/happo_cascade.yaml').read_text())
        cls.physical = yaml.safe_load((ROOT / 'harl/configs/envs_cfgs/cascade_reservoir.yaml').read_text())
        cls.args['train'].update(episode_length=4, n_rollout_threads=2)
        cls.args['model'].update(hidden_sizes=[8, 8], action_bins=7)
        cls.args['algo'].update(ppo_epoch=2, critic_epoch=2)
        cls.args['experiment'].update(episodes=2, seeds=[1], diagnostics=False)
        cls.data = CascadeData(ROOT / 'data/cascade_reservoir_new', cls.physical,
                               cls.args['experiment'])
        cls.windows = select_periodic_windows(cls.data, episode_days=4)

    def assert_state_equal(self, expected, actual):
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(expected, actual))
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(expected, actual)
        elif isinstance(expected, dict):
            self.assertEqual(set(expected), set(actual))
            for key in expected:
                self.assert_state_equal(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                self.assert_state_equal(left, right)
        else:
            self.assertEqual(expected, actual)

    @staticmethod
    def training_state(runner):
        return copy.deepcopy(dict(
            actors=[actor.actor.state_dict() for actor in runner.actor],
            actor_optimizers=[actor.actor_optimizer.state_dict() for actor in runner.actor],
            critic=runner.critic.critic.state_dict(),
            critic_optimizer=runner.critic.critic_optimizer.state_dict(),
            normalization=runner.value_normalizer.state_dict(),
            random_state=torch.get_rng_state(),
        ))

    def test_real_update_tracing_checkpoint_and_replay(self):
        from harl.envs.cascade_reservoir.checkpoint import load_checkpoint, ReplayTraceWriter
        from harl.envs.cascade_reservoir.trajectory_diagnostics import (
            TrajectoryDiagnosticsWriter, TRAJECTORY_DIAGNOSTIC_FIELDS)

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            directory = Path(directory)
            baseline = CascadeReservoirRunner(self.data, self.physical, self.args, 1,
                                               periodic_windows=self.windows)
            self.addCleanup(baseline.close)
            initial = self.training_state(baseline)
            baseline.run(2)
            expected = self.training_state(baseline)
            self.assertTrue(any(not torch.equal(initial['critic'][name], expected['critic'][name])
                                for name in initial['critic']))
            self.assertTrue(any(not torch.equal(initial['actors'][i][name], expected['actors'][i][name])
                                for i in range(5) for name in initial['actors'][i]))

            trajectory = directory / 'trajectory.csv'
            with TrajectoryDiagnosticsWriter(trajectory) as writer:
                traced = CascadeReservoirRunner(self.data, self.physical, self.args, 1,
                    periodic_windows=self.windows, trajectory_log=writer,
                    checkpoint_dir=directory / 'checkpoints')
                self.addCleanup(traced.close)
                traced.run(2)
            self.assert_state_equal(expected, self.training_state(traced))
            with trajectory.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4 * 2 * 5)
            self.assertEqual(set(rows[0]), set(TRAJECTORY_DIAGNOSTIC_FIELDS))
            self.assertEqual(len(traced.checkpoint_history), 1)

            checkpoint = directory / 'checkpoints/update_000001.pt'
            self.assertTrue(checkpoint.is_file())
            random_before_load = torch.get_rng_state().clone()
            policy = load_checkpoint(checkpoint)
            self.assertTrue(torch.equal(random_before_load, torch.get_rng_state()))
            self.assert_state_equal(traced.critic.critic.state_dict(), policy.critic.state_dict())
            self.assert_state_equal(traced.value_normalizer.state_dict(),
                                    policy.value_normalizer.state_dict())
            output = io.StringIO()
            replay = policy.replay([1965], writer=ReplayTraceWriter(output))
            reference = evaluate_policy(traced.actor, self.data, self.physical,
                seed=1, episode=2, training_steps=8, gamma=self.args['algo']['gamma'],
                source_years=[1965], episode_days=4, diagnostics=True)
            for field in ('energy_MWh', 'episode_reward', 'episode_mixed_rewards',
                          'episode_system_reward', 'spill_volume_m3'):
                np.testing.assert_allclose(replay[field], reference[field], rtol=0., atol=1e-10)
            self.assertEqual(len(list(csv.DictReader(io.StringIO(output.getvalue())))), 4 * 5)

    def test_cli_help_imports_complete_workflows(self):
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        for script, option in (('train_cascade.py', '--trajectory-diagnostics'),
                               ('replay_cascade_checkpoint.py', '--override-action')):
            with self.subTest(script=script):
                result = subprocess.run([sys.executable, '-B', str(ROOT / 'examples' / script), '--help'],
                    cwd=str(ROOT), env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(option, result.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
