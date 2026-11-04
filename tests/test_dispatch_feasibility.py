"""Independent full-physics checks of investigation-only dispatch attribution."""
import copy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'tests'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.dispatch_diagnostics.feasibility import analyze_transition, DispatchDiagnosticError
from harl.envs.cascade_reservoir.data_loader import CascadeData, IDS
from harl.envs.cascade_reservoir.episodes import episode_environment
from harl.envs.cascade_reservoir.reservoir_physics import ReservoirPhysics, InfeasibleCascade
from harl.envs.cascade_reservoir.xld_diagnostics import XLDDispatchDiagnostics
from test_xld_dispatch_diagnostics import synthetic_dispatch


def brute_grid(env, row, reservoir):
    actions = row['sampled_actions'].astype(float)/(env.action_bins-1)
    results = []
    for index in range(env.action_bins):
        candidate = actions.copy()
        candidate[reservoir] = index/(env.action_bins-1)
        results.append(env.physics.transition(row['storage_start'], row['external'],
            candidate, row['storage_lower_m3'], row['storage_upper_m3']))
    return results


def synthetic(i=2, **kwargs):
    """Move the independently specified XLD fixture to any cascade station."""
    ecology = kwargs.pop('ecology', 0.)
    env, row = synthetic_dispatch(**kwargs)
    if i != 2:
        order = np.arange(5)
        order[i], order[2] = order[2], order[i]
        data = env.data
        for name in ('vmin', 'vmax', 'release_cap', 'turbine_cap', 'power_cap', 'head'):
            setattr(data, name, getattr(data, name)[order].copy())
        original_anchors = data.anchors.copy()
        data.anchors = {name:original_anchors[IDS[order[j]]].copy() for j,name in enumerate(IDS)}
        data.levels = lambda values: np.asarray([
            np.interp(values[j], data.anchors[name][:, 1], data.anchors[name][:, 0]) for j,name in enumerate(IDS)])
        if data.head_sensitivity is not None:
            data.head_sensitivity = {key:value[order].copy() for key,value in data.head_sensitivity.items()}
        for name in ('guaranteed_output_mw', 'ecological_flow_m3s'):
            setattr(data.service_requirements, name, getattr(data.service_requirements, name)[order].copy())
        physics = ReservoirPhysics(data, env.physics.config)
        storage, external, actions = (row[name][order].copy() for name in ('storage_start', 'external', 'sampled_actions'))
        row = physics.transition(storage, external, actions/(env.action_bins-1), data.vmin, data.vmax)
        row.update(date='2000-01-01', sampled_actions=actions, action_bins=env.action_bins,
                   normalized_actions=row['actions'].copy())
        row['actions'] = np.where(row['controls'] > 0., actions, 0)
        env.physics = physics
        env.storage = row['storage_end'].copy()
        env.last_transition = row
    env.data.service_requirements.ecological_flow_m3s[i] = ecology
    return env, row


class DispatchFeasibilityTest(unittest.TestCase):
    def test_guarantee_action_costs_all_stations_match_full_physics(self):
        config = yaml.safe_load((ROOT/'harl/configs/envs_cfgs/cascade_reservoir.yaml').read_text())
        experiment = yaml.safe_load((ROOT/'harl/configs/algos_cfgs/happo_cascade.yaml').read_text())['experiment']
        rng = np.random.default_rng(413)
        for head in ('fixed', 'storage_sensitivity'):
            data = CascadeData(ROOT/'data/cascade_reservoir_new', dict(config, head_model=head), experiment)
            env = episode_environment(data, dict(config, head_model=head), [1965], days=3, action_bins=11)
            for _ in range(3):
                env.step(rng.integers(env.action_bins, size=5))
                row = env.last_transition
                before = copy.deepcopy(row)
                storage = env.storage.copy()
                numpy_state, python_state = np.random.get_state(), random.getstate()
                alternatives = [brute_grid(env, row, i) for i in range(5)]
                powers = np.array([[r['power'][i] for r in alternatives[i]] for i in range(5)])
                releases = np.array([[r['release'][i] for r in alternatives[i]] for i in range(5)])
                for thresholds in (data.service_requirements.guaranteed_output_mw,
                                   np.quantile(powers, .5, axis=1),
                                   powers[:, 0]+1e-5, powers[:, -1]+1e-5):
                    costs = env.physics.guaranteed_output_action_costs(row, env.action_bins, thresholds)
                    expected = ((np.asarray(thresholds)[:, None] > 0.) &
                                (np.asarray(thresholds)[:, None]-powers >
                                 1e-10*np.maximum(np.asarray(thresholds)[:, None], 1.)))
                    self.assertEqual(costs.shape, (5, env.action_bins))
                    self.assertEqual(costs.dtype, np.float32)
                    np.testing.assert_array_equal(costs, expected.astype(np.float32))
                thresholds = data.service_requirements.guaranteed_output_mw
                ecology = data.service_requirements.ecological_flow_m3s
                joint = env.physics.guaranteed_output_action_costs(
                    row, env.action_bins, thresholds, ecological_thresholds=ecology)
                expected = ((thresholds[:, None]-powers > 1e-10*np.maximum(thresholds[:, None], 1.)) |
                            (ecology[:, None]-releases > 1e-10*np.maximum(ecology[:, None], 1.)))
                np.testing.assert_array_equal(joint, expected.astype(np.float32))
                for key, value in row.items():
                    if isinstance(value, np.ndarray):
                        np.testing.assert_array_equal(value, before[key])
                np.testing.assert_array_equal(env.storage, storage)
                self.assertIs(env.last_transition, row)
                self.assertEqual(random.getstate(), python_state)
                np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])

    def test_guarantee_action_costs_nonmonotone_caps_and_forced(self):
        for i in range(5):
            for bins in (2, 11):
                with self.subTest(reservoir=IDS[i], bins=bins):
                    env, row = synthetic(i, head_model='storage_sensitivity', external=100.,
                                         guarantee=110., bins=bins)
                    thresholds = env.data.service_requirements.guaranteed_output_mw
                    costs = env.physics.guaranteed_output_action_costs(row, bins, thresholds)
                    powers = np.array([r['power'][i] for r in brute_grid(env, row, i)])
                    np.testing.assert_array_equal(costs[i], (110.-powers > 110.e-10).astype(np.float32))
                    self.assertEqual(costs[i, 0], 1.)
                    self.assertEqual(costs[i, -1], 1.)
                    self.assertEqual(bool((costs[i] == 0.).any()), bins > 2)
                    np.testing.assert_array_equal(np.delete(costs, i, axis=0), 0.)
            for head, turbine, installed in (('fixed', 12., 1e6), ('fixed', 10000., 8.),
                                             ('storage_sensitivity', 130., 1e6)):
                env, original = synthetic(i, head_model=head, external=100., guarantee=10.,
                                          turbine=turbine, bins=11)
                env.data.power_cap[i] = installed
                env.physics = ReservoirPhysics(env.data, env.physics.config)
                row = env.physics.transition(original['storage_start'], original['external'],
                    original['sampled_actions']/10., env.data.vmin, env.data.vmax)
                row['sampled_actions'] = original['sampled_actions']
                powers = np.array([r['power'][i] for r in brute_grid(env, row, i)])
                thresholds = env.data.service_requirements.guaranteed_output_mw
                costs = env.physics.guaranteed_output_action_costs(row, 11, thresholds)
                np.testing.assert_array_equal(costs[i], (10.-powers > 10.e-10).astype(np.float32))
                self.assertLessEqual(float(powers.max()), installed+1e-7)
            env, row = synthetic(i, available_water=0., guarantee=10., bins=11)
            self.assertEqual(row['controls'][i], 0.)
            costs = env.physics.guaranteed_output_action_costs(
                row, 11, env.data.service_requirements.guaranteed_output_mw)
            np.testing.assert_array_equal(costs[i], np.ones(11, dtype=np.float32))

    def test_guarantee_action_costs_tolerance_validation_and_anchor_order(self):
        env, row = synthetic_dispatch(available_water=0., external=20., guarantee=10., bins=11)
        # An equality next-state bound forces a positive release at every index.
        lower, upper = env.data.vmin.copy(), env.data.vmax.copy()
        lower[2] = upper[2] = row['storage_start'][2]
        forced = env.physics.transition(row['storage_start'], row['external'],
            row['sampled_actions']/10., lower, upper)
        self.assertEqual(forced['controls'][2], 0.)
        power = forced['power'][2]
        for relative, expected in ((.5e-10, 0.), (2.e-10, 1.)):
            thresholds = np.zeros(5)
            thresholds[2] = power/(1.-relative)
            costs = env.physics.guaranteed_output_action_costs(forced, 11, thresholds)
            np.testing.assert_array_equal(costs[2], np.full(11, expected, dtype=np.float32))
        env, row = synthetic_dispatch(head_model='storage_sensitivity', external=100., bins=11)
        thresholds = env.data.service_requirements.guaranteed_output_mw
        expected = env.physics.guaranteed_output_action_costs(row, 11, thresholds)
        env.data.anchors = dict(reversed(list(env.data.anchors.items())))
        np.testing.assert_array_equal(env.physics.guaranteed_output_action_costs(row, 11, thresholds), expected)
        for bins in (True, 1, 2.5):
            with self.assertRaises(ValueError):
                env.physics.guaranteed_output_action_costs(row, bins, thresholds)
        for invalid in (np.full(5, -1.), np.full(5, np.nan), np.full(5, np.inf), np.zeros(4)):
            with self.assertRaises(ValueError):
                env.physics.guaranteed_output_action_costs(row, 11, invalid)
        for name, value in (('inflow', -1.), ('low', -1.), ('high', np.inf),
                            ('release', np.nan), ('controls', .5)):
            invalid = copy.deepcopy(row)
            invalid[name][0] = value
            with self.assertRaises(ValueError):
                env.physics.guaranteed_output_action_costs(invalid, 11, thresholds)
        invalid = copy.deepcopy(row)
        invalid['inflow'][0] = 1e8
        with self.assertRaises(InfeasibleCascade):
            env.physics.guaranteed_output_action_costs(invalid, 11, thresholds)

    def test_guarantee_action_costs_optional_water_fraction_and_inactive_rows(self):
        env, row = synthetic_dispatch(head_model='storage_sensitivity', external=100.,
                                      guarantee=110., bins=11)
        thresholds = env.data.service_requirements.guaranteed_output_mw
        original = copy.deepcopy(row)
        binary = env.physics.guaranteed_output_action_costs(row, 11, thresholds)
        np.testing.assert_array_equal(
            env.physics.guaranteed_output_action_costs(row, 11, thresholds, water_weight=0.), binary)
        weighted = env.physics.guaranteed_output_action_costs(row, 11, thresholds, water_weight=.1)
        fraction = (np.arange(11, dtype=np.float64)/10.)**2
        np.testing.assert_allclose(weighted[2], binary[2]+.1*fraction, rtol=1e-7, atol=1e-8)
        self.assertEqual(weighted.dtype, np.float32)
        self.assertEqual(weighted.shape, (5, 11))
        self.assertEqual(float(weighted[2, 0]), float(binary[2, 0]))
        self.assertAlmostEqual(float(weighted[2, -1]-binary[2, -1]), .1, places=6)
        meeting = np.flatnonzero(binary[2] == 0.)
        self.assertGreater(len(meeting), 1)
        self.assertLess(weighted[2, meeting[0]], weighted[2, meeting[-1]])
        np.testing.assert_array_equal(np.delete(weighted, 2, axis=0), 0.)
        np.testing.assert_array_equal(
            env.physics.guaranteed_output_action_costs(row, 11, np.zeros(5), water_weight=.1), 0.)
        for key, value in row.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, original[key])
        for invalid in (-.1, np.nan, np.inf, True, None, 1e100):
            with self.assertRaises(ValueError):
                env.physics.guaranteed_output_action_costs(row, 11, thresholds, water_weight=invalid)

    def test_guarantee_action_costs_water_all_bad_and_forced_cases(self):
        for i in range(5):
            for head in ('fixed', 'storage_sensitivity'):
                env, row = synthetic(i, head_model=head, external=100., guarantee=1e6, bins=11)
                thresholds = env.data.service_requirements.guaranteed_output_mw
                binary = env.physics.guaranteed_output_action_costs(row, 11, thresholds)
                np.testing.assert_array_equal(binary[i], 1.)
                weighted = env.physics.guaranteed_output_action_costs(row, 11, thresholds, water_weight=.1)
                np.testing.assert_allclose(weighted[i], 1.+.1*(np.arange(11)/10.)**2, rtol=1e-7)
                self.assertEqual(int(np.argmin(weighted[i])), 0)
            env, row = synthetic(i, available_water=0., guarantee=10., bins=11)
            thresholds = env.data.service_requirements.guaranteed_output_mw
            np.testing.assert_array_equal(
                env.physics.guaranteed_output_action_costs(row, 11, thresholds, water_weight=.1),
                env.physics.guaranteed_output_action_costs(row, 11, thresholds))
        env, row = synthetic_dispatch(available_water=0., external=20., bins=11)
        lower, upper = env.data.vmin.copy(), env.data.vmax.copy()
        lower[2] = upper[2] = row['storage_start'][2]
        forced = env.physics.transition(row['storage_start'], row['external'],
            row['sampled_actions']/10., lower, upper)
        self.assertGreater(forced['release'][2], 0.)
        self.assertEqual(forced['controls'][2], 0.)
        thresholds = np.full(5, 1e6)
        np.testing.assert_array_equal(
            env.physics.guaranteed_output_action_costs(forced, 11, thresholds, water_weight=.1)[2], 1.)

    def test_joint_service_costs_high_head_thr_and_minimum_meeting(self):
        config = yaml.safe_load((ROOT/'harl/configs/envs_cfgs/cascade_reservoir.yaml').read_text())
        config = dict(config, head_model='storage_sensitivity')
        data = CascadeData(ROOT/'data/cascade_reservoir_new', config)
        physics = ReservoirPhysics(data, config)
        commands = np.zeros(5, dtype=int)
        external = np.zeros(5)
        external[4] = 2375.
        row = physics.transition(data.vmax.copy(), external, commands.astype(float), data.vmin, data.vmax)
        row['sampled_actions'] = commands
        env = SimpleNamespace(physics=physics, action_bins=100)
        guarantee, ecology = np.zeros(5), np.zeros(5)
        guarantee[4], ecology[4] = 1996., 2400.
        self.assertGreaterEqual(row['power'][4], guarantee[4])
        self.assertLess(row['release'][4], ecology[4])
        default = physics.guaranteed_output_action_costs(row, 100, guarantee)
        np.testing.assert_array_equal(
            physics.guaranteed_output_action_costs(row, 100, guarantee, ecological_thresholds=np.zeros(5)), default)
        self.assertEqual(default[4, 0], 0.)
        joint = physics.guaranteed_output_action_costs(row, 100, guarantee, ecological_thresholds=ecology)
        alternatives = brute_grid(env, row, 4)
        expected = np.array([(guarantee[4]-r['power'][4] > 1e-10*guarantee[4]) or
                             (ecology[4]-r['release'][4] > 1e-10*ecology[4]) for r in alternatives])
        np.testing.assert_array_equal(joint[4], expected.astype(np.float32))
        self.assertEqual(joint[4, 0], 1.)
        first_joint = int(np.flatnonzero(~expected)[0])
        self.assertGreater(first_joint, 0)
        self.assertGreaterEqual(alternatives[first_joint]['release'][4], ecology[4])
        weighted = physics.guaranteed_output_action_costs(
            row, 100, guarantee, water_weight=.1, ecological_thresholds=ecology)
        self.assertEqual(int(np.argmin(weighted[4])), first_joint)
        np.testing.assert_array_equal(weighted[:4], 0.)

    def test_joint_service_costs_ecology_only_total_release_and_forced_tolerance(self):
        env, row = synthetic(4, available_water=30., external=100., guarantee=0., turbine=1., bins=11)
        guarantee, ecology = np.zeros(5), np.zeros(5)
        ecology[4] = 50.
        costs = env.physics.guaranteed_output_action_costs(
            row, 11, guarantee, water_weight=.1, ecological_thresholds=ecology)
        alternatives = brute_grid(env, row, 4)
        expected = np.array([ecology[4]-r['release'][4] > 1e-10*ecology[4] for r in alternatives])
        np.testing.assert_allclose(costs[4], expected+.1*(np.arange(11)/10.)**2, rtol=1e-7)
        self.assertGreater(alternatives[-1]['release'][4], ecology[4])
        self.assertLess(alternatives[-1]['turbine'][4], ecology[4])
        self.assertAlmostEqual(float(costs[4, -1]), .1, places=6)
        self.assertEqual(int(np.argmin(costs[4])), int(np.flatnonzero(~expected)[0]))
        np.testing.assert_array_equal(costs[:4], 0.)
        env, row = synthetic_dispatch(available_water=0., external=20., bins=11)
        lower, upper = env.data.vmin.copy(), env.data.vmax.copy()
        lower[2] = upper[2] = row['storage_start'][2]
        forced = env.physics.transition(row['storage_start'], row['external'],
            row['sampled_actions']/10., lower, upper)
        for relative, expected in ((.5e-10, 0.), (2.e-10, 1.)):
            ecology = np.zeros(5)
            ecology[2] = forced['release'][2]/(1.-relative)
            costs = env.physics.guaranteed_output_action_costs(
                forced, 11, guarantee, water_weight=.1, ecological_thresholds=ecology)
            np.testing.assert_array_equal(costs[2], np.full(11, expected, dtype=np.float32))
        for invalid in (np.full(5, -1.), np.full(5, np.nan), np.full(5, np.inf), np.zeros(4)):
            with self.assertRaises(ValueError):
                env.physics.guaranteed_output_action_costs(row, 11, guarantee, ecological_thresholds=invalid)

    def test_historical_all_five_exact_grid_against_full_cascade(self):
        config = yaml.safe_load((ROOT/'harl/configs/envs_cfgs/cascade_reservoir.yaml').read_text())
        experiment = yaml.safe_load((ROOT/'harl/configs/algos_cfgs/happo_cascade.yaml').read_text())['experiment']
        for head in ('fixed', 'storage_sensitivity'):
            physical = dict(config, head_model=head)
            data = CascadeData(ROOT/'data/cascade_reservoir_new', physical, experiment)
            env = episode_environment(data, physical, [1965], days=4, action_bins=21)
            # Diagnostic-only thresholds above actual capacities make every
            # grid search observable; no source arrays or executed rewards change.
            env.operation_reward.requirements = SimpleNamespace(
                guaranteed_output_mw=np.full(5, 1e6), ecological_flow_m3s=np.full(5, 1e6))
            for day in range(4):
                env.step(np.full(5, 20 if day == 0 else 0))
                row = env.last_transition
                records = analyze_transition(env, row)
                self.assertEqual([r['reservoir_id'] for r in records], list(IDS))
                for i, record in enumerate(records):
                    expected = brute_grid(env, row, i)
                    for term, source in (('guaranteed_output', 'power'), ('ecological_flow', 'release')):
                        values = [float(r[source][i]) for r in expected]
                        self.assertEqual(record[term+'_best_grid_action_index'], int(np.argmax(values)))
                        np.testing.assert_allclose(record[term+'_best_grid_actual'], max(values), rtol=1e-10, atol=1e-7)
                        self.assertEqual(record[term+'_meeting_grid_action_count'], 0)
                    self.assertEqual(record['ecological_flow_classification'], 'same_state_capacity_limit')
                json.dumps(records, allow_nan=False)

    def test_xld_equivalence_for_each_classification(self):
        cases = [dict(guarantee=10.), dict(guarantee=100.),
                 dict(guarantee=100., available_water=1000., outlet=30.),
                 dict(guarantee=10., turbine=1.), dict(available_water=0.),
                 dict(head_model='storage_sensitivity', external=100., guarantee=110., bins=2),
                 dict(head_model='storage_sensitivity', external=100., guarantee=114., bins=200),
                 dict(guarantee=0.)]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                env, row = synthetic_dispatch(**kwargs)
                old = XLDDispatchDiagnostics(env, seed=1, evaluation_round=1,
                    window_id='equivalence', phase='training_evaluation').update(row, 1)
                new = analyze_transition(env, row)[2]
                self.assertEqual(new['guaranteed_output_classification'], old['classification'])
                if old['violating']:
                    self.assertEqual(new['guaranteed_output_best_grid_action_index'], old['best_grid_action_index'])
                    self.assertAlmostEqual(new['guaranteed_output_best_grid_actual'], old['best_grid_power_mw'], places=8)
                    self.assertAlmostEqual(new['guaranteed_output_continuous_max_actual'], old['continuous_max_power_mw'], places=8)
                    self.assertEqual(new['guaranteed_output_minimum_meeting_action_index'], old['minimum_meeting_action_index'])

    def test_all_station_nonmonotone_grid_gap_and_forced_cases(self):
        for i in range(5):
            with self.subTest(reservoir=IDS[i]):
                env, row = synthetic(i, head_model='storage_sensitivity', external=100.,
                    guarantee=110., bins=2, ecology=150.)
                record = analyze_transition(env, row)[i]
                self.assertEqual(record['guaranteed_output_classification'], 'discrete_grid_gap')
                self.assertEqual(record['ecological_flow_classification'], 'same_day_policy_alternative')
                self.assertAlmostEqual(record['guaranteed_output_continuous_best_release_m3s'], 160., places=8)
                self.assertAlmostEqual(record['guaranteed_output_continuous_max_actual'], .008829*160*80, places=8)
                self.assertLess(record['guaranteed_output_continuous_minimum_meeting_release_m3s'], 160.)
                self.assertAlmostEqual(record['ecological_flow_continuous_minimum_meeting_release_m3s'], 150., places=6)
                self.assertEqual(record['ecological_flow_minimum_meeting_action_index'], 1)
                self.assertAlmostEqual(record['ecological_flow_meeting_end_storage_delta_m3'], -100.*86400.)
                env, row = synthetic(i, available_water=0., ecology=1.)
                record = analyze_transition(env, row)[i]
                self.assertTrue(record['forced_action'])
                self.assertEqual(record['guaranteed_output_classification'], 'same_state_water_limit')
                self.assertEqual(record['ecological_flow_classification'], 'same_state_water_limit')
                self.assertEqual(record['ecological_flow_continuous_max_actual'], 0.)

    def test_original_downstream_sampled_action_and_observational_state(self):
        env, row = synthetic_dispatch(downstream_empty=True)
        env.data.service_requirements.ecological_flow_m3s[2] = 10.
        original, storage = copy.deepcopy(row), env.storage.copy()
        numpy_state, python_state = np.random.get_state(), random.getstate()
        with mock.patch.object(env.physics, 'transition', wraps=env.physics.transition) as replay:
            records = analyze_transition(env, row)
        self.assertGreater(replay.call_count, 0)
        for call in replay.call_args_list:
            # Only XLD has an active/violated service in this fixture.
            np.testing.assert_array_equal(call.args[2][[0,1,3,4]], row['sampled_actions'][[0,1,3,4]]/(env.action_bins-1))
        for key in row:
            if isinstance(row[key], np.ndarray):
                np.testing.assert_array_equal(row[key], original[key])
        np.testing.assert_array_equal(env.storage, storage)
        self.assertIs(env.last_transition, row)
        self.assertEqual(env.index, 1)
        self.assertEqual(random.getstate(), python_state)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        self.assertEqual(records[2]['guaranteed_output_classification'], 'same_day_policy_alternative')
        with mock.patch.object(env.physics, 'transition', side_effect=RuntimeError('broken counterfactual')):
            with self.assertRaises(DispatchDiagnosticError):
                analyze_transition(env, row)
        invalid = copy.deepcopy(row)
        del invalid['sampled_actions']
        with self.assertRaises(DispatchDiagnosticError):
            analyze_transition(env, invalid)

    def test_ecological_dispatch_ceiling_and_monotone_storage_breaks(self):
        env, original = synthetic(0, available_water=1000., guarantee=0., ecology=100.)
        data = env.data
        data.release_cap[1] = 30.
        storage = original['storage_start'].copy()
        storage[1] = data.vmax[1]
        env.physics = ReservoirPhysics(data, env.physics.config)
        row = env.physics.transition(storage, original['external'],
            original['sampled_actions']/(env.action_bins-1), data.vmin, data.vmax)
        row.update(date='2000-01-01', sampled_actions=original['sampled_actions'])
        record = analyze_transition(env, row)[0]
        self.assertEqual(record['ecological_flow_classification'], 'same_state_dispatch_limit')
        self.assertEqual(record['ecological_flow_continuous_max_actual'], 30.)
        self.assertFalse(record['ecological_flow_water_limit_proven'])

        env, original = synthetic(2, head_model='storage_sensitivity', external=100.,
                                  guarantee=1e5, bins=31)
        # A genuine interpolation kink creates separate quadratic power pieces.
        env.data.anchors['XLD'] = np.array([[0., 0.], [35., 86400.*50.], [100., 86400.*100.]])
        env.physics = ReservoirPhysics(env.data, env.physics.config)
        row = env.physics.transition(original['storage_start'], original['external'],
            original['sampled_actions']/(env.action_bins-1), env.data.vmin, env.data.vmax)
        row.update(date='2000-01-01', sampled_actions=original['sampled_actions'])
        record = analyze_transition(env, row)[2]
        normalized = original['sampled_actions']/(env.action_bins-1)
        dense = []
        for action in np.linspace(0., 1., 1001):
            commands = normalized.copy()
            commands[2] = action
            dense.append(env.physics.transition(row['storage_start'], row['external'],
                commands, row['storage_lower_m3'], row['storage_upper_m3'])['power'][2])
        self.assertGreaterEqual(record['guaranteed_output_continuous_max_actual']+1e-7, max(dense))
        self.assertLess(record['guaranteed_output_continuous_max_actual']-max(dense), .01)


if __name__ == '__main__':
    unittest.main(verbosity=2)
