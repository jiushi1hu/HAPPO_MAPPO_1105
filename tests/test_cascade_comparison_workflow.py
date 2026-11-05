"""Matched algorithm dispatch, atomic output checks and a short real workflow."""
import copy
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import inspect
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples import train_cascade as cli
from examples import train_cascade_compare as compare
from examples import train_cascade_parallel as parallel
from harl.algorithms.actors.happo import HAPPO
from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.periodic_evaluation import select_periodic_windows
from harl.runners.cascade_mappo_runner import CascadeMAPPORunner
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
from harl.runners.on_policy_ha_runner import OnPolicyHARunner


def comparison_options(**changes):
    values = dict(algorithms=['happo', 'mappo'], episode_log=None, no_plot=True,
        check_data_only=False, xld_diagnostics=False, trajectory_diagnostics=False,
        save_checkpoints=False, skip_final_evaluation=True, plot_python=sys.executable)
    values.update(changes)
    return SimpleNamespace(**values)


class ComparisonDispatchTest(unittest.TestCase):
    def test_cli_defaults_to_both_algorithms_and_supports_single_selection(self):
        selections = [([], ['happo', 'mappo']),
                      (['--algorithms', 'mappo'], ['mappo']),
                      (['--algorithms', 'happo'], ['happo'])]
        for arguments, expected in selections:
            with self.subTest(algorithms=expected), ExitStack() as stack:
                data_factory = stack.enter_context(mock.patch.object(
                    cli, 'CascadeData', return_value=mock.sentinel.data))
                helper = stack.enter_context(mock.patch.object(compare, 'run_parallel_experiment',
                    side_effect=lambda *args, **kwargs: {'algorithm': kwargs['algorithm_name']}))
                # This test isolates CLI dispatch; complete histories and real
                # comparison validation are exercised by the integration test.
                stack.enter_context(mock.patch.object(compare, 'validate_comparison',
                                                       side_effect=lambda value: value))
                plots = stack.enter_context(mock.patch.object(compare.subprocess, 'run'))
                stack.enter_context(redirect_stdout(io.StringIO()))
                result = compare.main(arguments+['--no-plot', '--skip-final-evaluation'])
                self.assertEqual([row['algorithm'] for row in result['runs']], expected)
                self.assertEqual(helper.call_count, len(expected))
                data_factory.assert_called_once()
                plots.assert_not_called()
                for call, algorithm in zip(helper.call_args_list, expected):
                    self.assertIs(call.args[0], mock.sentinel.data)
                    self.assertIs(call.kwargs['runner_cls'], compare.RUNNERS[algorithm])
                    self.assertEqual(call.kwargs['algorithm_name'], algorithm)
                    self.assertIsNone(call.args[3].episode_log)
                    self.assertTrue(call.args[3].no_plot)

    def test_each_algorithm_receives_fresh_settings_and_separate_output_paths(self):
        config, physical = cli.load_configuration()
        expected_config, expected_physical = copy.deepcopy(config), copy.deepcopy(physical)
        entries = []

        def helper(data, received_physical, received_config, options, **kwargs):
            entries.append(dict(data=data, physical=received_physical, config=received_config,
                options=options, before_physical=copy.deepcopy(received_physical),
                before_config=copy.deepcopy(received_config), **kwargs))
            # A mutation by one algorithm must not contaminate its competitor
            # or the caller's configuration, including nested lists/mappings.
            received_config['model']['hidden_sizes'][0] = -1
            received_physical['reward_penalties']['guaranteed_output_violation_weight'] = 999.
            return dict(algorithm=kwargs['algorithm_name'])

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)/'paired.jsonl'
            options = comparison_options(episode_log=base)
            with mock.patch.object(compare, 'run_parallel_experiment', side_effect=helper), \
                    mock.patch.object(compare, 'validate_comparison'), \
                    redirect_stdout(io.StringIO()):
                compare.run_comparison(mock.sentinel.data, physical, config, options)
            self.assertEqual([entry['before_config'] for entry in entries],
                             [expected_config, expected_config])
            self.assertEqual([entry['before_physical'] for entry in entries],
                             [expected_physical, expected_physical])
            self.assertIsNot(entries[0]['config'], entries[1]['config'])
            self.assertIsNot(entries[0]['physical'], entries[1]['physical'])
            self.assertIsNot(entries[0]['options'], entries[1]['options'])
            for entry, algorithm, runner_class in zip(entries, ('happo', 'mappo'),
                    (CascadeReservoirRunner, CascadeMAPPORunner)):
                self.assertIs(entry['runner_cls'], runner_class)
                self.assertEqual(entry['options'].episode_log,
                                 base.with_name('paired_'+algorithm+'.jsonl'))
                self.assertIsNot(entry['options'], options)
            self.assertEqual(options.episode_log, base)
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertEqual(config, expected_config)
        self.assertEqual(physical, expected_physical)

    def test_second_algorithm_collision_is_rejected_before_first_training(self):
        config, physical = cli.load_configuration()
        config['experiment']['seeds'] = [1, 2, 3]
        for kind in ('summary', 'steps', 'checkpoints'):
            with self.subTest(output=kind), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                base = directory/'paired.jsonl'
                second = directory/'paired_mappo.jsonl'
                if kind == 'summary':
                    collision = second
                elif kind == 'steps':
                    collision = directory/'paired_mappo_seed2_steps.csv'
                else:
                    collision = directory/'paired_mappo_seed2_checkpoints'
                if kind == 'checkpoints':
                    collision.mkdir()
                else:
                    collision.write_text('pre-existing output', encoding='utf-8')
                options = comparison_options(episode_log=base, save_checkpoints=True)
                with mock.patch.object(compare, 'run_parallel_experiment') as helper:
                    with self.assertRaisesRegex(ValueError, 'already exists'):
                        compare.run_comparison(mock.sentinel.data, physical, config, options)
                    helper.assert_not_called()
                self.assertEqual(list(directory.iterdir()), [collision])
                if kind != 'checkpoints':
                    self.assertEqual(collision.read_text(encoding='utf-8'), 'pre-existing output')

    def test_joint_kl_is_rejected_by_cli_before_loading_data_or_training(self):
        for arguments in ([], ['--algorithms', 'mappo']):
            with self.subTest(arguments=arguments), \
                    mock.patch.object(cli, 'CascadeData') as data_factory, \
                    mock.patch.object(compare, 'run_parallel_experiment') as helper, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
                with self.assertRaises(SystemExit) as raised:
                    compare.main(arguments+['--no-plot', '--target-kl', '.01',
                                            '--kl-rollback', '--joint-target-kl', '.03'])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn('does not support joint_target_kl', errors.getvalue())
                data_factory.assert_not_called()
                helper.assert_not_called()

    def test_original_entry_point_and_happo_training_dispatch_remain_defaults(self):
        self.assertIs(CascadeReservoirRunner.actor_class, HAPPO)
        self.assertEqual(CascadeReservoirRunner.algorithm_name, 'happo')
        self.assertIs(CascadeReservoirRunner.train, OnPolicyHARunner.train)
        self.assertIs(inspect.signature(cli.main).parameters['experiment_runner'].default,
                      parallel.run_parallel_experiment)
        defaults = inspect.signature(parallel.run_parallel_experiment).parameters
        self.assertIs(defaults['runner_cls'].default, CascadeReservoirRunner)
        self.assertIsNone(defaults['algorithm_name'].default)
        with mock.patch.object(cli, 'CascadeData', return_value=mock.sentinel.data), \
                redirect_stdout(io.StringIO()):
            helper = mock.Mock(return_value=mock.sentinel.result)
            result = cli.main(['--no-plot'], experiment_runner=helper)
        self.assertIs(result, mock.sentinel.result)
        helper.assert_called_once()
        self.assertFalse(hasattr(helper.call_args.args[3], 'algorithms'))


class ShortComparisonIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.physical = cli.load_configuration()
        cls.config['train'].update(episode_length=4, n_rollout_threads=2)
        cls.config['model'].update(hidden_sizes=[8, 8], action_bins=7)
        cls.config['algo'].update(ppo_epoch=1, critic_epoch=1, target_kl=None,
                                 kl_rollback=False, joint_target_kl=None)
        cls.config['experiment'].update(episodes=2, seeds=[1, 2, 3], diagnostics=False,
            trajectory_diagnostics=False, save_checkpoints=False, checkpoint_interval=0,
            xld_diagnostics=False)
        cls.data = CascadeData(ROOT/'data/cascade_reservoir_new', cls.physical,
                               cls.config['experiment'])
        cls.windows = select_periodic_windows(cls.data, episode_days=4)

    def test_three_paired_seeds_return_complete_histories_without_run_files(self):
        config_before, physical_before = copy.deepcopy(self.config), copy.deepcopy(self.physical)
        options = comparison_options()
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            directory = Path(directory)
            stack.enter_context(mock.patch.object(parallel, 'select_periodic_windows',
                                                  return_value=self.windows))
            stack.enter_context(mock.patch.object(parallel, 'PERIODIC_EVALUATION_DAYS', 4))
            plots = stack.enter_context(mock.patch.object(compare.subprocess, 'run',
                side_effect=AssertionError('No-plot workflow launched a plotting process')))
            stack.enter_context(redirect_stdout(io.StringIO()))
            os.chdir(directory)
            try:
                result = compare.run_comparison(self.data, self.physical, self.config, options)
            finally:
                os.chdir(original_directory)
            self.assertEqual(list(directory.iterdir()), [])
            self.assertFalse((directory/'results').exists())
            plots.assert_not_called()
        self.assertEqual(self.config, config_before)
        self.assertEqual(self.physical, physical_before)
        self.assertEqual([row['algorithm'] for row in result['runs']], ['happo', 'mappo'])
        happo, mappo = [row['payload'] for row in result['runs']]
        self.assertEqual(happo['comparison_context'], mappo['comparison_context'])
        context = happo['comparison_context']
        self.assertTrue(context['data_hashes'])
        settings = context['shared_settings']
        self.assertEqual(settings['seeds'], [1, 2, 3])
        self.assertEqual(settings['episode_horizon_days'], 4)
        self.assertEqual(settings['parallel_environments'], 2)
        self.assertEqual(settings['episodes_per_seed'], 2)
        self.assertEqual(settings['periodic_windows'], self.windows)
        for key in ('device', 'train', 'model', 'algo'):
            self.assertEqual(settings[key], self.config[key])
        for algorithm, payload in (('happo', happo), ('mappo', mappo)):
            with self.subTest(algorithm=algorithm):
                self.assertEqual(payload['algorithm'], algorithm)
                self.assertEqual(payload['seeds'], [1, 2, 3])
                self.assertEqual(len(payload['histories']), 3)
                self.assertEqual(len(payload['evaluation_histories']), 3)
                for seed, history, evaluations in zip(payload['seeds'], payload['histories'],
                                                     payload['evaluation_histories']):
                    self.assertEqual(len(history), 1)
                    self.assertEqual(history[0]['seed'], seed)
                    self.assertEqual(history[0]['training_steps'], 8)
                    self.assertEqual(history[0]['completed_episodes'], 2)
                    self.assertEqual([row['evaluation_round'] for row in evaluations], [0, 1])
                    self.assertTrue(all(row['status'] == 'ok' for row in evaluations))
                    self.assertTrue(all(row['seed'] == seed for row in evaluations))
        for left, right in zip(happo['histories'], mappo['histories']):
            self.assertEqual(left[0]['window_start_year'], right[0]['window_start_year'])
        # Returned results must be ordinary detached data, not live workers,
        # tensors, optimizer objects, or mutable views into runner histories.
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main(verbosity=2)
