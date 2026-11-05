"""Inference persistence, strict provenance and isolated deterministic replay."""
import copy
import csv
import inspect
import io
from pathlib import Path
import random
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harl.envs.cascade_reservoir.checkpoint import (
    REPLAY_TRACE_FIELDS, ReplayTraceWriter, load_checkpoint, save_checkpoint, _digest, _state_hash)
from harl.envs.cascade_reservoir.data_loader import CascadeData, IDS
from harl.envs.cascade_reservoir.evaluation import evaluate_policy
from harl.envs.cascade_reservoir.periodic_evaluation import select_periodic_windows
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
from harl.runners.cascade_mappo_runner import CascadeMAPPORunner


class CascadeCheckpointTest(unittest.TestCase):
    runner_class = CascadeReservoirRunner
    @classmethod
    def setUpClass(cls):
        with (ROOT/'harl/configs/envs_cfgs/cascade_reservoir.yaml').open() as stream:
            cls.physical = yaml.safe_load(stream)
        with (ROOT/'harl/configs/algos_cfgs/happo_cascade.yaml').open() as stream:
            cls.args = yaml.safe_load(stream)
        cls.args['train'].update(episode_length=8, n_rollout_threads=2)
        cls.args['experiment'].update(episodes=2, diagnostics=False)
        cls.data = CascadeData(ROOT/'data/cascade_reservoir_new', cls.physical, cls.args['experiment'])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'update_000001.pt'
        self.runner = self.runner_class(self.data, self.physical, self.args, 1,
            periodic_windows=select_periodic_windows(self.data, episode_days=8))
        self.addCleanup(self.runner.close)
        # Persistence tests need a completed-policy marker; the separate workflow
        # integration test exercises actual optimizer updates and runner saving.
        self.runner.update_count = 1
        self.runner.completed_episodes = 2
        self.runner.training_steps = 16
        self.runner.value_normalizer.update(np.arange(60, dtype=np.float32).reshape(10, 6))

    def payload(self):
        kwargs = dict(map_location='cpu')
        if 'weights_only' in inspect.signature(torch.load).parameters:
            kwargs['weights_only'] = True
        return torch.load(self.path, **kwargs)

    def test_roundtrip_frozen_models_and_exclusive_save(self):
        before = torch.get_rng_state().clone()
        info = save_checkpoint(self.runner, self.path)
        policy = load_checkpoint(self.path)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(info['episode_horizon_days'], 8)
        self.assertEqual(len(info['sha256']), 64)
        for original, frozen in zip(self.runner.actor, policy.actors):
            for name, value in original.actor.state_dict().items():
                self.assertTrue(torch.equal(value, frozen.actor.state_dict()[name]))
            self.assertFalse(frozen.actor.training)
            self.assertTrue(all(not p.requires_grad for p in frozen.actor.parameters()))
        for name, value in self.runner.critic.critic.state_dict().items():
            self.assertTrue(torch.equal(value, policy.critic.state_dict()[name]))
        for name, value in self.runner.value_normalizer.state_dict().items():
            self.assertTrue(torch.equal(value, policy.value_normalizer.state_dict()[name]))
        content = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            save_checkpoint(self.runner, self.path)
        self.assertEqual(content, self.path.read_bytes())
        self.assertNotIn('optimizer', self.payload())
        if not self.runner.algo_args['algo'].get('share_param', False):
            self.assertEqual(len(self.payload()['actors']), 5)
            self.assertNotIn('actor_parameter_sharing', self.payload()['metadata'])

    def test_legacy_independent_mappo_format_remains_supported(self):
        if self.runner.algo_args['algo'].get('share_param', False):
            self.skipTest('This case exercises the older independent-actor format')
        self.runner.algorithm_name = 'mappo'
        save_checkpoint(self.runner, self.path)
        self.assertEqual(len(self.payload()['actors']), 5)
        policy = load_checkpoint(self.path)
        self.assertEqual(policy.metadata['algorithm_name'], 'mappo')
        self.assertEqual(len({id(actor.actor) for actor in policy.actors}), 5)
        self.assertTrue(policy.replay([1965])['feasible'])

    def test_replay_matches_existing_evaluator_and_csv_returns(self):
        save_checkpoint(self.runner, self.path)
        policy = load_checkpoint(self.path)
        stream = io.StringIO()
        numpy_before, random_before = np.random.get_state(), random.getstate()
        torch_before = torch.get_rng_state().clone()
        expected = evaluate_policy(self.runner.actor, self.data, self.physical,
            seed=1, episode=2, training_steps=16, gamma=self.args['algo']['gamma'],
            source_years=[1965], episode_days=8, diagnostics=True)
        result = policy.replay([1965], writer=ReplayTraceWriter(stream))
        for name in ('energy_MWh', 'episode_reward', 'episode_mixed_rewards',
                     'episode_system_reward', 'spill_volume_m3'):
            np.testing.assert_allclose(result[name], expected[name], rtol=0., atol=1e-10)
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        self.assertEqual(random_before, random.getstate())
        self.assertTrue(np.array_equal(numpy_before[1], np.random.get_state()[1]))
        rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
        self.assertEqual(len(rows), 40)
        self.assertEqual(set(rows[0]), set(REPLAY_TRACE_FIELDS))
        for i in range(5):
            self.assertEqual(rows[i]['reservoir'], IDS[i])
            own = rows[i::5]
            reward = np.asarray([float(r['mixed_reward']) for r in own])
            for t, row in enumerate(own):
                actual = float(row['realized_mixed_reward_to_go'])
                expected_return = sum(self.args['algo']['gamma']**j*r for j, r in enumerate(reward[t:]))
                self.assertAlmostEqual(actual, expected_return, places=12)
            self.assertEqual(float(own[-1]['next_value']), 0.)
        again = io.StringIO()
        self.assertEqual(result, policy.replay([1965], writer=ReplayTraceWriter(again)))
        self.assertEqual(stream.getvalue(), again.getvalue())

    def test_single_override_and_later_reactive_policy(self):
        save_checkpoint(self.runner, self.path)
        policy = load_checkpoint(self.path)
        baseline_stream = io.StringIO()
        policy.replay([1965], writer=ReplayTraceWriter(baseline_stream))
        baseline = list(csv.DictReader(io.StringIO(baseline_stream.getvalue())))
        original = int(baseline[1]['policy_action_index'])
        changed = (original+50) % self.runner.action_bins
        stream = io.StringIO()
        result = policy.replay([1965], writer=ReplayTraceWriter(stream),
                               action_overrides={(1, 'BHT'): changed})
        rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
        self.assertEqual(int(rows[1]['command_action_index']), changed)
        self.assertEqual(int(rows[1]['policy_action_index']), original)
        self.assertEqual(sum(int(r['action_overridden']) for r in rows), 1)
        self.assertEqual(result['action_overrides'], [dict(episode_step=1, reservoir='BHT', action_index=changed)])
        self.assertTrue(all(r['command_action_index'] == r['policy_action_index'] for r in rows[5:]))
        self.assertNotEqual(rows[1]['storage_end_m3'], baseline[1]['storage_end_m3'])

    def test_invalid_horizon_split_years_and_overrides(self):
        save_checkpoint(self.runner, self.path)
        policy = load_checkpoint(self.path)
        for years, days, overrides in (
            ([1965], 9, None), ([1965, 1998], None, None),
            ([1965, 1965], None, None), ([1965.0], None, None),
            ([1964], None, None), ([], None, None),
            ([1965], None, {(0, 'BHT'): 1}), ([1965], None, {(9, 'BHT'): 1}),
            ([1965], None, {(1, 'BHT'): 100}), ([1965], None, {(1, 'UNKNOWN'): 1}),
            ([1965], None, {(True, 'BHT'): 1}), ([1965], None, {(1, 'BHT'): 1.5})):
            with self.subTest(years=years, days=days, overrides=overrides), self.assertRaises(ValueError):
                policy.replay(years, days, action_overrides=overrides)
        self.assertEqual(policy.replay([1998])['phase'], 'evaluation')

    def test_source_metadata_and_model_mismatches_are_rejected(self):
        save_checkpoint(self.runner, self.path)
        with mock.patch('harl.envs.cascade_reservoir.checkpoint._source_hashes', return_value={}):
            with self.assertRaisesRegex(ValueError, 'source/config'):
                load_checkpoint(self.path)
        original = self.payload()
        changed = copy.deepcopy(original)
        changed['metadata']['episode_horizon_days'] = 9
        torch.save(changed, self.path)
        with self.assertRaisesRegex(ValueError, 'metadata hash'):
            load_checkpoint(self.path)
        changed = copy.deepcopy(original)
        next(iter(changed['actors'][0].values())).add_(1.)
        torch.save(changed, self.path)
        with self.assertRaisesRegex(ValueError, 'model state hash'):
            load_checkpoint(self.path)
        changed = copy.deepcopy(original)
        next(iter(changed['critic'].values())).fill_(float('nan'))
        torch.save(changed, self.path)
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            load_checkpoint(self.path)

    def test_relocated_data_must_match_and_no_normalizer_mode(self):
        save_checkpoint(self.runner, self.path)
        relocated = Path(self.tmp.name)/'data'
        shutil.copytree(self.data.root, relocated)
        policy = load_checkpoint(self.path, data_root=relocated)
        self.assertEqual(policy.data.root, relocated.resolve())
        target = relocated/'reservoir_parameters.csv'
        with target.open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'dataset hash'):
            load_checkpoint(self.path, data_root=relocated)
        self.runner.value_normalizer = None
        self.runner.algo_args['train']['use_valuenorm'] = False
        path = Path(self.tmp.name)/'unscaled.pt'
        save_checkpoint(self.runner, path)
        self.assertIsNone(load_checkpoint(path).value_normalizer)

    def test_writer_validates_schema_and_finite_values(self):
        writer = ReplayTraceWriter(io.StringIO())
        with self.assertRaises(ValueError):
            writer.write({})
        row = dict.fromkeys(REPLAY_TRACE_FIELDS, 0.)
        row['value'] = float('inf')
        with self.assertRaises(ValueError):
            writer.write(row)


class SharedCascadeCheckpointTest(CascadeCheckpointTest):
    runner_class = CascadeMAPPORunner

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.args = copy.deepcopy(cls.args)
        cls.args['algo']['share_param'] = True

    def test_one_state_one_frozen_network_and_five_identity_views(self):
        save_checkpoint(self.runner, self.path)
        stored = self.payload()
        self.assertEqual(len(stored['actors']), 1)
        self.assertEqual(len(stored['metadata']['model_hashes']['actors']), 1)
        self.assertTrue(stored['metadata']['actor_parameter_sharing'])
        self.assertEqual(stored['metadata']['agent_ids'], list(range(5)))
        self.assertEqual(stored['metadata']['raw_actor_observation_dim'], 15)
        self.assertEqual(stored['metadata']['actor_observation_dim'], 20)
        before = torch.get_rng_state().clone()
        numpy_before, random_before = np.random.get_state(), random.getstate()
        frozen = load_checkpoint(self.path)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(random_before, random.getstate())
        self.assertTrue(np.array_equal(numpy_before[1], np.random.get_state()[1]))
        self.assertEqual(len({id(actor.actor) for actor in frozen.actors}), 1)
        self.assertEqual(len({id(actor.core) for actor in frozen.actors}), 1)
        self.assertEqual([actor.agent_id for actor in frozen.actors], list(range(5)))
        self.assertFalse(hasattr(frozen.actors[0].core, 'actor_optimizer'))
        first_linear = next(module for module in frozen.actors[0].actor.modules()
                            if isinstance(module, torch.nn.Linear))
        self.assertEqual(first_linear.in_features, 20)
        observation = self.runner.envs[0].actor_observations()[0:1]
        for original, view in zip(self.runner.actor, frozen.actors):
            with torch.no_grad():
                expected = original.evaluate_actions(observation, np.zeros((1, 1), dtype=int))[2]
                actual = view.evaluate_actions(observation, np.zeros((1, 1), dtype=int))[2]
            np.testing.assert_array_equal(expected.probs.cpu().numpy(), actual.probs.cpu().numpy())

    def test_shared_architecture_metadata_is_validated(self):
        save_checkpoint(self.runner, self.path)
        original = self.payload()
        for change in ('ids', 'input', 'states'):
            with self.subTest(change=change):
                stored = copy.deepcopy(original)
                if change == 'ids':
                    stored['metadata']['agent_ids'] = [0, 0, 2, 3, 4]
                elif change == 'input':
                    stored['metadata']['actor_observation_dim'] = 15
                else:
                    stored['actors'] = stored['actors']*5
                    stored['metadata']['model_hashes']['actors'] *= 5
                stored['metadata_sha256'] = _digest(stored['metadata'])
                torch.save(stored, self.path)
                with self.assertRaises(ValueError):
                    load_checkpoint(self.path)

    def test_shared_model_failure_restores_rng_state(self):
        save_checkpoint(self.runner, self.path)
        stored = self.payload()
        name = next(iter(stored['actors'][0]))
        stored['actors'][0][name] = stored['actors'][0][name][:-1].clone()
        stored['metadata']['model_hashes']['actors'] = [_state_hash(stored['actors'][0])]
        stored['metadata_sha256'] = _digest(stored['metadata'])
        torch.save(stored, self.path)
        before = torch.get_rng_state().clone()
        with self.assertRaisesRegex(ValueError, 'Malformed'):
            load_checkpoint(self.path)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
