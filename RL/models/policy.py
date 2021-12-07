import numpy as np
import torch
import torch.nn as nn

from RL.models.utils import ValueFunctionNormaliser

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class SettlersAgentPolicy(nn.Module):
    def __init__(self, observation_module, action_head_module, lstm_in_dim=128, lstm_layers=1, lstm_size=64,
                 value_mlp_size=64, value_normalisation=True):
        super(SettlersAgentPolicy, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.lstm_layers = lstm_layers
        self.lstm_size = lstm_size

        self.use_value_normalisation = value_normalisation
        if value_normalisation:
            self.value_normaliser = ValueFunctionNormaliser(mean=125.0, std=125.0)

        self.policy_type = "neural_network"

        self.standard_obs_keys = ["proposed_trade", "current_resources", "current_player_main", "next_player_main",
                                  "next_next_player_main", "next_next_next_player_main"]
        self.list_int_obs_keys = ["current_player_played_dev", "current_player_hidden_dev",
                                  "next_player_played_dev", "next_next_player_played_dev",
                                  "next_next_next_player_played_dev"]

        self.observation_module = observation_module

        self.lstm = nn.LSTM(num_layers=lstm_layers, hidden_size=lstm_size, input_size=lstm_in_dim, batch_first=False)

        self.action_head_module = action_head_module

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.value_network = nn.Sequential(
            init_(nn.Linear(lstm_size + lstm_in_dim, value_mlp_size)),
            nn.Tanh(),
            init_(nn.Linear(value_mlp_size, 1))
        )

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    def base(self, obs_dict, hidden_states, done_masks):
        lstm_input = self.observation_module(obs_dict)
        lstm_output, hidden_states = self._forward_lstm(lstm_input, hidden_states, done_masks)

        main_output = torch.cat((lstm_input, lstm_output), dim=-1)

        value = self.value_network(main_output)
        return value, main_output, hidden_states

    def act(self, obs_dict, hidden_states, nonterminal_masks, action_masks, deterministic=False,
            return_entropy=False, condition_on_action_type=None):
        custom_inputs = {
            "current_resources": obs_dict["current_resources"],
            "proposed_trade": obs_dict["proposed_trade"]
        }

        value, main_out, hidden_states = self.base(obs_dict, hidden_states, nonterminal_masks)

        actions, action_log_probs, entropy = self.action_head_module(
            main_input=main_out, masks=action_masks, custom_inputs=custom_inputs,
            deterministic=deterministic, condition_on_action_type=condition_on_action_type
        )

        if return_entropy:
            return value, actions, action_log_probs, hidden_states, entropy
        else:
            return value, actions, action_log_probs, hidden_states

    def evaluate_actions(self, obs_dict, hidden_states, nonterminal_masks, actions, action_masks):
        custom_inputs = {
            "current_resources": obs_dict["current_resources"],
            "proposed_trade": obs_dict["proposed_trade"]
        }

        value, main_out, hidden_states = self.base(obs_dict, hidden_states, nonterminal_masks)

        _, action_log_probs, entropys = self.action_head_module(
            main_input=main_out, masks=action_masks, custom_inputs=custom_inputs,
            actions=actions
        )

        return value, action_log_probs, entropys, hidden_states

    def get_value(self, obs_dict, hidden_states, nonterminal_masks):
        value, _, _ = self.base(obs_dict, hidden_states, nonterminal_masks)
        return value

    def _forward_lstm(self, x, hidden_states, nonterminal_masks):
        """don't think this works for >1 layer LSTM atm..."""
        cxs = hidden_states[1]
        hxs = hidden_states[0]
        if x.size(0) == hxs.size(0):
            x, hxs = self.lstm(x.unsqueeze(0), ((hxs*nonterminal_masks).unsqueeze(0),
                                                (cxs*nonterminal_masks).unsqueeze(0)))
            cxs = hxs[1].squeeze(0)
            hxs = hxs[0].squeeze(0)
            hidden_states = (hxs, cxs)
            x = x.squeeze(0)
        else:
            #x is (B, N, -1) flattened to (B*N, -1)
            B = hxs.size(0)
            T = int(x.size(0) / B)

            x = x.view(T, B, x.size(1))

            nonterminal_masks = nonterminal_masks.view(T, B)

            has_zeros = ((nonterminal_masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu())
            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()
            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            cxs = cxs.unsqueeze(0)

            outputs = []
            for i in range(len(has_zeros)-1):
                #process steps with no masks together - a lot faster with GPU
                start_idx = has_zeros[i]
                end_idx = has_zeros[i+1]
                rnn_scores, hidden_states = self.lstm(
                    x[start_idx:end_idx], ((hxs * nonterminal_masks[start_idx].view(1, -1, 1)),
                                           (cxs * nonterminal_masks[start_idx].view(1, -1, 1)))
                )
                cxs = hidden_states[1]
                hxs = hidden_states[0]

                outputs.append(rnn_scores)

            x = torch.cat(outputs, dim=0)
            x = x.view(T*B, -1)
            hxs = hxs.squeeze(0)
            cxs = cxs.squeeze(0)
            hidden_states = (hxs, cxs)

        return x, hidden_states

    def obs_to_torch(self, obs):
        for key in self.standard_obs_keys:
            obs[key] = torch.tensor(obs[key], dtype=torch.float32, device=self.dummy_param.device)
            if len(obs[key].shape) == 1:
                obs[key] = obs[key].unsqueeze(0)
        for key in self.list_int_obs_keys:
            if isinstance(obs[key][0], list):
                for k in range(len(obs[key])):
                    obs[key][k] = torch.tensor(obs[key][k], dtype=torch.long, device=self.dummy_param.device)
            else:
                obs[key] = [torch.tensor(obs[key], dtype=torch.long, device=self.dummy_param.device)]
        obs["tile_representations"] = torch.tensor(obs["tile_representations"], dtype=torch.float32,
                                                   device=self.dummy_param.device)
        if len(obs["tile_representations"].shape) == 2:
            obs["tile_representations"] = obs["tile_representations"].unsqueeze(0)
        return obs

    def act_masks_to_torch(self, masks):
        for z in range(len(masks)):
            masks[z] = torch.tensor(masks[z], dtype=torch.float32, device=self.dummy_param.device).unsqueeze(0)
            if z == 1 or z == 6 or z == 9:
                masks[z] = torch.transpose(masks[z], 0, 1)
        return masks

    def torch_act_to_np(self, action):
        for z in range(len(action)):
            if isinstance(action[z], list):
                for p in range(len(action[z])):
                    action[z][p] = action[z][p].squeeze().cpu().data.numpy()
            else:
                action[z] = action[z].squeeze().cpu().data.numpy()
        return action