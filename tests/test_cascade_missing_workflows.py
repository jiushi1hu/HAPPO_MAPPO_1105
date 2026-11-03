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
import unittest

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.evaluation import evaluate_policy
from harl.envs.cascade_reservoir.periodic_evaluation import select_periodic_windows
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner


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
