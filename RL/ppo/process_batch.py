import itertools
import torch
import numpy as np

from torch.nn.utils.rnn import pad_sequence
from RL.ppo.utils import _flatten_helper

OBS_KEYS = ["proposed_trade", "current_resources", "tile_representations", "current_player_main",
            "current_player_played_dev", "current_player_hidden_dev", "next_player_main", "next_player_played_dev",
            "next_next_player_main", "next_next_player_played_dev", "next_next_next_player_main",
            "next_next_next_player_played_dev"]
OBS_TYPES = ["normal", "normal", "normal", "normal", "list", "list", "normal", "list", "normal", "list", "normal",
             "list"]
TYPE_CONDITIONAL_MASKS = [1, 6, 9]

class BatchProcessor(object):
    def __init__(self, args, lstm_dim, obs_keys=OBS_KEYS, obs_type=OBS_TYPES,
                 type_conditional_masks=TYPE_CONDITIONAL_MASKS, num_action_heads=11, device="cuda"):
        self.args = args
        self.num_steps = args.num_steps
        self.num_parallel = args.num_processes * args.num_envs_per_process

        self.obs_keys = obs_keys
        self.obs_type = obs_type
        self.type_conditional_masks = type_conditional_masks
        self.lstm_dim = lstm_dim
        self.num_action_heads = num_action_heads

        self.games_complete = 0

        self.device = device

    def process_rollouts(self, rollouts):
        rollouts = list(zip(*rollouts))

        obs = [inner for outer in rollouts[0] for inner in outer]
        self.obs_dict = {}
        for i, key in enumerate(self.obs_keys):
            if self.obs_type[i] == "list":
                p1 = [[obs[k][t][key][0] for k in range(self.num_parallel)] for t in range(self.num_steps+1)]
                p2 = list(itertools.chain.from_iterable(p1))
                p3 = pad_sequence(p2, batch_first=True)
                self.obs_dict[key] = p3.view(self.num_steps + 1, self.num_parallel, p3.shape[1]).to(self.device)
            else:
                self.obs_dict[key] = torch.stack(
                    [torch.vstack([obs[k][t][key] for k in range(self.num_parallel)]) for t in range(self.num_steps+1)]
                ).to(self.device)

        hidden_states = [inner for outer in rollouts[1] for inner in outer]
        self.hidden_states = (
            torch.stack([torch.vstack([hidden_states[k][t][0] for k in range(self.num_parallel)]) for t in
                         range(self.num_steps+1)]).to(self.device),
            torch.stack([torch.vstack([hidden_states[k][t][1] for k in range(self.num_parallel)]) for t in
                         range(self.num_steps + 1)]).to(self.device)
        )

        rewards = [inner for outer in rollouts[2] for inner in outer]
        self.rewards = torch.stack([
            torch.vstack([torch.tensor(rewards[k][t], dtype=torch.float32, device=self.device).view(1,1)
                          for k in range(self.num_parallel)]) for t in range(self.num_steps)
        ]).to(self.device)

        actions = [inner for outer in rollouts[3] for inner in outer]
        self.actions = []
        for i in range(self.num_action_heads):
            self.actions.append(
                torch.stack(
                    [torch.vstack([torch.tensor(np.array(actions[k][t][i]), dtype=torch.long, device=self.device).view(1,-1) \
                                   for k in range(self.num_parallel)]) for t in range(self.num_steps)]
                ).to(self.device)
            )

        action_masks = [inner for outer in rollouts[4] for inner in outer]
        self.action_masks = []
        for i in range(self.num_action_heads):
            if i in self.type_conditional_masks:
                self.action_masks.append(
                    torch.stack([torch.cat([action_masks[k][t][i] for k in range(self.num_parallel)], dim=1) for t in
                                 range(self.num_steps)], dim=1).to(self.device)
                )
            else:
                self.action_masks.append(
                    torch.stack(
                        [torch.vstack([action_masks[k][t][i] for k in range(self.num_parallel)]) for t in range(self.num_steps)]
                    ).to(self.device)
                )

        action_log_probs = [inner for outer in rollouts[5] for inner in outer]
        self.action_log_probs = torch.stack([
            torch.vstack([action_log_probs[k][t] for k in range(self.num_parallel)]) for t in range(self.num_steps)
        ]).to(self.device)

        masks = [inner for outer in rollouts[6] for inner in outer]
        self.masks = torch.stack([
           torch.vstack([
               torch.tensor(masks[k][t], dtype=torch.float32).view(1,1) for k in range(self.num_parallel)]) \
               for t in range(self.num_steps+1)
        ]).to(self.device)

        self.games_complete += int(torch.sum(1.0-self.masks).item())

    def compute_advantages(self, actor_critic):

        recurrent_hidden_states_in = (_flatten_helper(self.num_steps+1, self.num_parallel, self.hidden_states[0]),
                                   _flatten_helper(self.num_steps+1, self.num_parallel, self.hidden_states[1]))
        obs_dict_in = {}
        for key in self.obs_keys:
            obs_dict_in[key] = _flatten_helper(self.num_steps+1, self.num_parallel, self.obs_dict[key])
        masks_in = _flatten_helper(self.num_steps+1, self.num_parallel, self.masks)

        self.values = actor_critic.get_value(
            obs_dict_in, recurrent_hidden_states_in, masks_in
        ).view(self.num_steps + 1, self.num_parallel, -1)
        self.returns = torch.zeros_like(self.rewards, device=self.device)

        """Generalised advantage estimation"""
        gae = 0
        for step in reversed(range(self.num_steps)):
            delta = self.rewards[step] + self.args.gamma * self.values[step + 1] * self.masks[step + 1] - self.values[step]
            gae = delta + self.args.gamma * self.args.gae_lambda * self.masks[step + 1] *  gae
            self.returns[step] = gae + self.values[step]

        advantages = self.returns - self.values[:-1]
        self.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    def generator(self, num_mini_batch):
        assert self.num_parallel >= num_mini_batch
        num_envs_per_batch = self.num_parallel // num_mini_batch
        perm = torch.randperm(self.num_parallel)

        for start_ind in range(0, self.num_parallel, num_envs_per_batch):
            obs_dict_batch = {}
            for key in self.obs_keys:
                obs_dict_batch[key] = []

            recurrent_hidden_state_batch = [[], []]
            actions_batch = [[] for _ in range(self.num_action_heads)]
            action_masks_batch = [[] for _ in range(self.num_action_heads)]
            value_preds_batch = []
            returns_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targets = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                for key in self.obs_keys:
                    obs_dict_batch[key].append(self.obs_dict[key][:-1, ind])

                recurrent_hidden_state_batch[0].append(self.hidden_states[0][0:1, ind])
                recurrent_hidden_state_batch[1].append(self.hidden_states[1][0:1, ind])

                for i in range(self.num_action_heads):
                    actions_batch[i].append(self.actions[i][:, ind])
                    if i in self.type_conditional_masks:
                        action_masks_batch[i].append(self.action_masks[i][:, :, ind])
                    else:
                        action_masks_batch[i].append(self.action_masks[i][:, ind])

                value_preds_batch.append(self.values[:-1, ind])
                returns_batch.append(self.returns[:, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targets.append(self.advantages[:, ind])

            #all tensors of size (num_steps, num_parallel, -1)
            for key in self.obs_keys:
                obs_dict_batch[key] = torch.stack(obs_dict_batch[key], 1)

            recurrent_hidden_state_batch[0] = torch.stack(recurrent_hidden_state_batch[0], 1).view(num_envs_per_batch, -1)
            recurrent_hidden_state_batch[1] = torch.stack(recurrent_hidden_state_batch[1], 1).view(num_envs_per_batch, -1)

            for i in range(self.num_action_heads):
                actions_batch[i] = torch.stack(actions_batch[i], 1)
                if i in self.type_conditional_masks:
                    action_masks_batch[i] = torch.stack(action_masks_batch[i], 2)
                else:
                    action_masks_batch[i] = torch.stack(action_masks_batch[i], 1)

            value_preds_batch = torch.stack(value_preds_batch, 1)
            returns_batch = torch.stack(returns_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targets = torch.stack(adv_targets, 1)

            #flatten
            for key in self.obs_keys:
                obs_dict_batch[key] = _flatten_helper(self.num_steps, num_envs_per_batch, obs_dict_batch[key])

            for i in range(self.num_action_heads):
                actions_batch[i] = _flatten_helper(self.num_steps, num_envs_per_batch, actions_batch[i])
                if i in self.type_conditional_masks:
                    num_types = action_masks_batch[i].shape[0]
                    T, N = action_masks_batch[i].shape[1], action_masks_batch[i].shape[2]
                    action_masks_batch[i] = action_masks_batch[i].view(num_types, T*N, -1)
                else:
                    action_masks_batch[i] = _flatten_helper(self.num_steps, num_envs_per_batch, action_masks_batch[i])

            value_preds_batch = _flatten_helper(self.num_steps, num_envs_per_batch, value_preds_batch)
            returns_batch = _flatten_helper(self.num_steps, num_envs_per_batch, returns_batch)
            masks_batch = _flatten_helper(self.num_steps, num_envs_per_batch, masks_batch)
            old_action_log_probs_batch = _flatten_helper(self.num_steps, num_envs_per_batch, old_action_log_probs_batch)
            adv_targets = _flatten_helper(self.num_steps, num_envs_per_batch, adv_targets)

            yield obs_dict_batch, recurrent_hidden_state_batch, actions_batch, action_masks_batch, value_preds_batch, \
                  returns_batch, masks_batch, old_action_log_probs_batch, adv_targets