"""MAPPO using the unchanged cascade environment and collection lifecycle."""
from harl.algorithms.actors.mappo import MAPPO
from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
from harl.runners.on_policy_ma_runner import OnPolicyMARunner


class CascadeMAPPORunner(CascadeReservoirRunner):
    """Five independent MAPPO actors and the shared six-output value model."""

    actor_class = MAPPO
    algorithm_name = 'mappo'

    def train(self):
        """Dispatch the MAPPO update while reusing the cascade workflow."""
        return OnPolicyMARunner.train(self)
