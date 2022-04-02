import itertools
import torch
import numpy as np

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence

from RL.ppo.utils import _flatten_helper, _flatten_helper_reshape

OBS_KEYS = ["proposed_trade", "current_resources", "tile_representations", "current_player_main",
            "current_player_played_dev", "current_player_hidden_dev", "next_player_main", "next_player_played_dev",
            "next_next_player_main", "next_next_player_played_dev", "next_next_next_player_main",
            "next_next_next_player_played_dev"]
OBS_TYPES = ["normal", "normal", "normal", "normal", "list", "list", "normal", "list", "normal", "list", "normal",
             "list"]
TYPE_CONDITIONAL_MASKS = [1, 6, 9]

class BatchProcessor(object):
    def __init__(self, args, lstm_dim, obs_keys=OBS_KEYS, obs_type=OBS_TYPES,
                 type_conditional_masks=TYPE_CONDITIONAL_MASKS, num_action_heads=12, device="cuda",
                 stored_device="cpu"):
        self.args = args
        self.num_steps = args.num_steps
        self.num_parallel = args.num_processes * args.num_envs_per_process

        self.obs_keys = obs_keys
        self.obs_type = obs_type
        self.type_conditional_masks = type_conditional_masks
        self.lstm_dim = lstm_dim
        self.num_action_heads = num_action_heads

        self.games_complete = 0

        self.stored_device = stored_device
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
                self.obs_dict[key] = p3.view(self.num_steps + 1, self.num_parallel, p3.shape[1]).to(self.stored_device)
            else:
                self.obs_dict[key] = torch.stack(
                    [torch.cat([obs[k][t][key] for k in range(self.num_parallel)], dim=0) for t in range(self.num_steps+1)]
                ).to(self.stored_device)

        hidden_states = [inner for outer in rollouts[1] for inner in outer]
        self.hidden_states = (
            torch.stack([torch.cat([hidden_states[k][t][0] for k in range(self.num_parallel)], dim=0) for t in
                         range(self.num_steps+1)]).to(self.stored_device),
            torch.stack([torch.cat([hidden_states[k][t][1] for k in range(self.num_parallel)], dim=0) for t in
                         range(self.num_steps + 1)]).to(self.stored_device)
        )

        rewards = [inner for outer in rollouts[2] for inner in outer]
        self.rewards = torch.stack([
            torch.cat([torch.tensor(rewards[k][t], dtype=torch.float32, device=self.stored_device).view(1,1)
                          for k in range(self.num_parallel)], dim=0) for t in range(self.num_steps)
        ]).to(self.stored_device)

        actions = [inner for outer in rollouts[3] for inner in outer]
        self.actions = []
        for i in range(self.num_action_heads):
            self.actions.append(
                torch.stack(
                    [torch.cat([torch.tensor(np.array(actions[k][t][i]), dtype=torch.long, device=self.stored_device).view(1,-1) \
                                   for k in range(self.num_parallel)], dim=0) for t in range(self.num_steps)]
                ).to(self.stored_device)
            )

        action_masks = [inner for outer in rollouts[4] for inner in outer]
        self.action_masks = []
        for i in range(self.num_action_heads):
            if i in self.type_conditional_masks:
                self.action_masks.append(
                    torch.stack([torch.cat([action_masks[k][t][i] for k in range(self.num_parallel)], dim=1) for t in
                                 range(self.num_steps)], dim=1).to(self.stored_device)
                )
            else:
                self.action_masks.append(
                    torch.stack(
                        [torch.cat([action_masks[k][t][i] for k in range(self.num_parallel)], dim=0) for t in range(self.num_steps)]
                    ).to(self.stored_device)
                )

        action_log_probs = [inner for outer in rollouts[5] for inner in outer]
        self.action_log_probs = torch.stack([
            torch.cat([action_log_probs[k][t] for k in range(self.num_parallel)], dim=0) for t in range(self.num_steps)
        ]).to(self.stored_device)

        masks = [inner for outer in rollouts[6] for inner in outer]
        self.masks = torch.stack([
           torch.cat([
               torch.tensor(masks[k][t], dtype=torch.float32).view(1,1) for k in range(self.num_parallel)], dim=0) \
               for t in range(self.num_steps+1)
        ]).to(self.stored_device)

        self.games_complete += int(torch.sum(1.0-self.masks).item())

    def compute_advantages_alt(self, actor_critic, max_processes_at_once=10):
        """break down the computation - stop GPU running out of memory."""
        self.values = torch.zeros(self.num_steps+1, self.num_parallel, 1).to(self.stored_device)
        self.returns = torch.zeros_like(self.rewards).to(self.stored_device)

        start_inds = np.arange(0, self.num_parallel, max_processes_at_once)
        end_inds = np.minimum(start_inds + max_processes_at_once, self.num_parallel)
        for i in range(len(start_inds)):
            num_proc = end_inds[i] - start_inds[i]
            if actor_critic.include_lstm:
                recurrent_hidden_states_in = (_flatten_helper_reshape(self.num_steps+1, num_proc, self.hidden_states[0][:, start_inds[i]:end_inds[i], ...]).to(self.device),
                                              _flatten_helper_reshape(self.num_steps+1, num_proc, self.hidden_states[1][:, start_inds[i]:end_inds[i], ...]).to(self.device))
            else:
                recurrent_hidden_states_in = None

            obs_dict_in = {}
            for key in self.obs_keys:
                obs_dict_in[key] = _flatten_helper_reshape(self.num_steps+1, num_proc, self.obs_dict[key][:, start_inds[i]:end_inds[i], ...]).to(self.device)

            masks_in = _flatten_helper_reshape(self.num_steps+1, num_proc, self.masks[:, start_inds[i]:end_inds[i], ...]).to(self.device)

            self.values[:, start_inds[i]:end_inds[i], :] = actor_critic.get_value(
                obs_dict_in, recurrent_hidden_states_in, masks_in
            ).reshape(self.num_steps + 1, num_proc, -1).to(self.stored_device)

        if actor_critic.use_value_normalisation:
            self.values = actor_critic.value_normaliser.denormalise(self.values)

        """Generalised advantage estimation"""
        gae = 0
        for step in reversed(range(self.num_steps)):
            delta = self.rewards[step] + self.args.gamma * self.values[step + 1] * self.masks[step + 1] - self.values[step]
            gae = delta + self.args.gamma * self.args.gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + self.values[step]

        advantages = self.returns - self.values[:-1]
        self.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)


    # def compute_advantages(self, actor_critic):
    #
    #     recurrent_hidden_states_in = (_flatten_helper(self.num_steps+1, self.num_parallel, self.hidden_states[0]),
    #                                _flatten_helper(self.num_steps+1, self.num_parallel, self.hidden_states[1]))
    #     obs_dict_in = {}
    #     for key in self.obs_keys:
    #         obs_dict_in[key] = _flatten_helper(self.num_steps+1, self.num_parallel, self.obs_dict[key])
    #     masks_in = _flatten_helper(self.num_steps+1, self.num_parallel, self.masks)
    #
    #     self.values = actor_critic.get_value(
    #         obs_dict_in, recurrent_hidden_states_in, masks_in
    #     ).view(self.num_steps + 1, self.num_parallel, -1)
    #     self.returns = torch.zeros_like(self.rewards, device=self.device)
    #
    #     """Generalised advantage estimation"""
    #     gae = 0
    #     for step in reversed(range(self.num_steps)):
    #         delta = self.rewards[step] + self.args.gamma * self.values[step + 1] * self.masks[step + 1] - self.values[step]
    #         gae = delta + self.args.gamma * self.args.gae_lambda * self.masks[step + 1] *  gae
    #         self.returns[step] = gae + self.values[step]
    #
    #     advantages = self.returns - self.values[:-1]
    #     self.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    def generator_standard(self, num_mini_batch):
        batch_size = self.num_steps * self.num_parallel
        mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True
        )

        for indices in sampler:
            obs_dict_batch = {}
            for key in self.obs_keys:
                obs_dict_batch[key] = self.obs_dict[key][:-1].view(-1, *self.obs_dict[key].size()[2:])[indices].to(self.device)
            recurrent_hidden_states_batch = None

            actions_batch = [[] for _ in range(self.num_action_heads)]
            action_masks_batch = [[] for _ in range(self.num_action_heads)]

            for i in range(self.num_action_heads):
                if i in self.type_conditional_masks:
                    action_masks_batch[i] = self.action_masks[i].view(self.action_masks[i].size()[0], -1, *self.action_masks[i].size()[3:])[:, indices, ...].to(self.device)
                else:
                    action_masks_batch[i] = self.action_masks[i].view(-1, *self.action_masks[i].size()[2:])[indices].to(self.device)
                actions_batch[i] = self.actions[i].view(-1, *self.actions[i].size()[2:])[indices].to(self.device)

            value_preds_batch = self.values[:-1].view(-1, 1)[indices].to(self.device)
            returns_batch = self.returns.view(-1, 1)[indices].to(self.device)
            masks_batch = self.masks[:-1].view(-1, 1)[indices].to(self.device)
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices].to(self.device)
            adv_targets = self.advantages.view(-1, 1)[indices].to(self.device)

            yield obs_dict_batch, recurrent_hidden_states_batch, actions_batch, action_masks_batch, value_preds_batch, \
                  returns_batch, masks_batch, old_action_log_probs_batch, adv_targets


    def generator_lstm(self, num_mini_batch, total_batch_size, truncated_seq_len):
        T = self.num_steps
        num_parallel = self.num_parallel
        assert T % truncated_seq_len == 0

        num_sequences_per_minibatch = total_batch_size // num_mini_batch // truncated_seq_len
        N = num_sequences_per_minibatch
        time_inds = [];
        process_inds = []
        for p_id in range(num_parallel):
            for t_s in range(0, T, truncated_seq_len):
                process_inds.append([p_id] * truncated_seq_len)
                time_inds.append(np.arange(t_s, t_s + truncated_seq_len))
        inds_permutation = np.random.permutation(len(time_inds))

        for start_ind in range(0, len(inds_permutation), num_sequences_per_minibatch):
            obs_dict_batch = {}
            for key in self.obs_keys:
                obs_dict_batch[key] = []

            recurrent_hidden_batch = [[], []]
            actions_batch = [[] for _ in range(self.num_action_heads)]
            action_masks_batch = [[] for _ in range(self.num_action_heads)]
            value_preds_batch = []
            returns_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targets = []

            for offset in range(num_sequences_per_minibatch):
                t_inds = time_inds[inds_permutation[start_ind + offset]]
                p_inds = process_inds[inds_permutation[start_ind + offset]]

                for key in self.obs_keys:
                    obs_dict_batch[key].append(self.obs_dict[key][t_inds, p_inds, ...])

                recurrent_hidden_batch[0].append(self.hidden_states[0][t_inds[0]:t_inds[1], p_inds[0], ...])
                recurrent_hidden_batch[1].append(self.hidden_states[1][t_inds[0]:t_inds[1], p_inds[0], ...])

                for i in range(self.num_action_heads):
                    actions_batch[i].append(self.actions[i][t_inds, p_inds, ...])
                    if i in self.type_conditional_masks:
                        action_masks_batch[i].append(self.action_masks[i][:, t_inds, p_inds, ...])
                    else:
                        action_masks_batch[i].append(self.action_masks[i][t_inds, p_inds, ...])

                value_preds_batch.append(self.values[t_inds, p_inds])
                returns_batch.append(self.returns[t_inds, p_inds])
                masks_batch.append(self.masks[t_inds, p_inds])
                old_action_log_probs_batch.append(self.action_log_probs[t_inds, p_inds])
                adv_targets.append(self.advantages[t_inds, p_inds])

            for key in self.obs_keys:
                obs_dict_batch[key] = torch.stack(obs_dict_batch[key], 1)

            recurrent_hidden_batch[0] = torch.stack(recurrent_hidden_batch[0], 1).view(N, -1).to(self.device)
            recurrent_hidden_batch[1] = torch.stack(recurrent_hidden_batch[1], 1).view(N, -1).to(self.device)

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

            # flatten
            for key in self.obs_keys:
                obs_dict_batch[key] = _flatten_helper(truncated_seq_len, N, obs_dict_batch[key]).to(self.device)

            for i in range(self.num_action_heads):
                actions_batch[i] = _flatten_helper(truncated_seq_len, N, actions_batch[i]).to(self.device)
                if i in self.type_conditional_masks:
                    num_types = action_masks_batch[i].shape[0]
                    T, N = action_masks_batch[i].shape[1], action_masks_batch[i].shape[2]
                    action_masks_batch[i] = action_masks_batch[i].view(num_types, T * N, -1).to(self.device)
                else:
                    action_masks_batch[i] = _flatten_helper(truncated_seq_len, N, action_masks_batch[i]).to(self.device)

            value_preds_batch = _flatten_helper(truncated_seq_len, N, value_preds_batch).to(self.device)
            returns_batch = _flatten_helper(truncated_seq_len, N, returns_batch).to(self.device)
            masks_batch = _flatten_helper(truncated_seq_len, N, masks_batch).to(self.device)
            old_action_log_probs_batch = _flatten_helper(truncated_seq_len, N, old_action_log_probs_batch).to(self.device)
            adv_targets = _flatten_helper(truncated_seq_len, N, adv_targets).to(self.device)

            yield obs_dict_batch, recurrent_hidden_batch, actions_batch, action_masks_batch, value_preds_batch, \
                  returns_batch, masks_batch, old_action_log_probs_batch, adv_targets