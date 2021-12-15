import torch
import torch.nn as nn

class PPO():
    def __init__(self, actor_critic, args):
        self.actor_critic = actor_critic

        self.args = args

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch

        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef_start

        self.max_grad_norm = args.max_grad_norm
        self.recompute_returns = args.recompute_returns

        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda

        self.optimiser = torch.optim.Adam(actor_critic.parameters(), lr=args.lr, eps=args.eps)

    def update(self, rollout_storage):
        value_loss_epoch = 0
        action_loss_epoch = 0
        entropy_loss_epoch = 0

        for e in range(self.ppo_epoch):
            with torch.no_grad():
                rollout_storage.compute_advantages_alt(self.actor_critic, 10)

            if self.actor_critic.include_lstm:
                total_batch_size = rollout_storage.num_parallel * rollout_storage.num_steps
                data_generator = rollout_storage.generator_lstm(num_mini_batch=self.num_mini_batch,
                                                               total_batch_size=total_batch_size,
                                                               truncated_seq_len=self.args.truncated_seq_len)
            else:
                data_generator = rollout_storage.generator_standard(self.num_mini_batch)

            for sample in data_generator:
                obs_dict_batch, recurrent_batch, actions_batch, action_masks_batch, value_preds_batch, returns_batch, \
                    masks_batch, old_action_log_probs_batch, adv_target = sample

                if self.actor_critic.use_value_normalisation:
                    value_preds_batch = self.actor_critic.value_normaliser.normalise(value_preds_batch)
                    returns_batch = self.actor_critic.value_normaliser.normalise(returns_batch)

                values, action_log_probs, entropy, _ = self.actor_critic.evaluate_actions(
                    obs_dict_batch, recurrent_batch, masks_batch, actions_batch, action_masks_batch
                )

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_target
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_target
                action_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = value_preds_batch + \
                                     (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - returns_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                self.optimiser.zero_grad()
                (value_loss * self.value_loss_coef + action_loss - entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimiser.step()

                value_loss_epoch += value_loss.item() * self.value_loss_coef
                action_loss_epoch += action_loss.item()
                entropy_loss_epoch += entropy.item() * self.entropy_coef

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        entropy_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, entropy_loss_epoch