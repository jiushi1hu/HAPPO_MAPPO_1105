"""Independent MAPPO updates on a common, frozen cascade rollout."""
import numpy as np

from harl.runners.on_policy_base_runner import OnPolicyBaseRunner


class OnPolicyMARunner(OnPolicyBaseRunner):
    """Update each independent actor without sequential likelihood corrections."""

    def train(self):
        """Use the same per-agent return channels and critic update as HAPPO."""
        if self.algo_args['algo'].get('share_param', False):
            raise ValueError('Cascade MAPPO requires independent actors')
        if self.algo_args['algo'].get('joint_target_kl') is not None:
            raise ValueError('MAPPO does not support joint_target_kl')
        self.joint_kl_info = None
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[:-1] - self.value_normalizer.denormalize(
                self.critic_buffer.value_preds[:-1])
        else:
            advantages = self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
        if advantages.shape[-1] not in (1, self.num_agents + 1):
            raise ValueError('Expected one legacy return or one mixed return per actor plus a legacy return')
        if self.algo_args['algo'].get('normalize_shared_advantages', False):
            advantages = ((advantages - advantages.mean(axis=(0, 1), keepdims=True)) /
                          (advantages.std(axis=(0, 1), keepdims=True) + 1e-5))

        actor_train_infos = []
        # Execution is serial, but each independent optimizer uses the original
        # rollout probabilities. There is no cross-agent gradient or F update.
        for agent_id in range(self.num_agents):
            buffer = self.actor_buffer[agent_id]
            # The shared buffer schema retains this slot. MAPPO validates the
            # unit placeholder and never includes it in its actor objective.
            buffer.update_factor(np.ones_like(buffer.active_masks[:-1]))
            channel = 0 if advantages.shape[-1] == 1 else agent_id
            options = {'diagnostics': True} if getattr(self, 'diagnostics_enabled', False) else {}
            info = self.actor[agent_id].train(
                buffer, advantages[..., channel:channel+1].copy(), 'EP', **options)
            # Preserve the existing trajectory/diagnostic schema; these values
            # certify that no sequential importance correction was applied.
            info['factor_min'] = info['factor_max'] = 1.
            actor_train_infos.append(info)

        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)
        return actor_train_infos, critic_train_info
