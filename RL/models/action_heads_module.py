import math
import numpy as np
import torch
import torch.nn as nn

from RL.distributions import Categorical, DiagGaussian

DEFAULT_MLP_SIZE = 64

class MultiActionHeadsGeneralised(nn.Module):
    def __init__(self, action_heads, autoregressive_map, main_input_dim, log_prob_masks={},
                 type_conditional_action_masks=None):
        super(MultiActionHeadsGeneralised, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.autoregressive_map = autoregressive_map
        self.main_input_dim = main_input_dim
        self.log_prob_masks = log_prob_masks
        self.type_conditional_action_masks = type_conditional_action_masks

        self.action_heads = nn.ModuleList()
        for head in action_heads:
            self.action_heads.append(head)

    def forward(self, main_input, masks, actions=None, custom_inputs=None, deterministic=False):
        head_outputs = []
        action_outputs = []
        head_log_probs_filtered = []

        joint_action_log_prob = 0
        entropy = 0

        for i, head in enumerate(self.action_heads):
            main_head_inputs = []
            for entry in self.autoregressive_map[i]:
                if entry[0] == -1:
                    initial_input = main_input
                else:
                    initial_input = head_outputs[entry[0]]
                if entry[1] is not None:
                    initial_input = entry[1](initial_input)
                if entry[0] >= 0:
                    initial_input *= (1-head_log_probs_filtered[entry[0]]) #filter inputs that are masked out.
                main_head_inputs.append(initial_input)
            main_head_inputs = torch.cat(main_head_inputs, dim=-1)

            #get relevant action masks
            if len(self.type_conditional_action_masks[i]):
                head_mask = 1
                for prev_head_ind, head_type_to_option_map in self.type_conditional_action_masks[i].items():
                    if actions is not None:
                        head_mask *= masks[i][head_type_to_option_map[actions[prev_head_ind]].squeeze(),
                                     np.arange(main_head_inputs.size(0)), :]
                    else:
                        head_mask *= masks[i][head_type_to_option_map[action_outputs[prev_head_ind]].squeeze(),
                                     np.arange(main_head_inputs.size(0)), :]
            else:
                head_mask = masks[i]

            if head.returns_distribution:
                head_distribution = head(main_head_inputs, head_mask, custom_inputs)

                if deterministic:
                    head_action = head_distribution.mode()
                else:
                    if head.type == "normal":
                        head_action = head_distribution.rsample()
                    else:

                        try:
                            head_action = head_distribution.sample()
                        except:
                            torch.save((head_distribution, main_head_inputs, head_mask, custom_inputs, i), "inside_head_error.pt")
                            print("successfully dumped inner info")

                if head.type == "categorical":
                    one_hot_head_action = torch.zeros(main_input.size(0), head.output_dim, device=self.dummy_param.device)
                    if actions is None:
                        one_hot_head_action.scatter_(-1, head_action, 1.0)
                    else:
                        one_hot_head_action.scatter_(-1, actions[i], 1.0)
                    head_outputs.append(one_hot_head_action)
                else:
                    head_outputs.append(head_action)
                action_outputs.append(head_action)

                if actions is None:
                    head_log_prob = head_distribution.log_probs(head_action)
                else:
                    head_log_prob = head_distribution.log_probs(actions[i])
                log_prob_mask = torch.ones(main_input.size(0), 1, dtype=torch.float32, device=self.dummy_param.device)
                if self.log_prob_masks[i] is not None:
                    for prev_head_ind, head_type_mask in self.log_prob_masks[i].items():
                        filter = head_log_probs_filtered[prev_head_ind] #allows us to ignore the mask from this head if its log prob has been filtered by an earlier head.
                        if actions is None:
                            acts_to_mask = action_outputs
                        else:
                            acts_to_mask = actions
                        head_prob_mask = head_type_mask[acts_to_mask[prev_head_ind].squeeze()].view(-1, 1)
                        log_prob_mask *= ((1-filter) * head_prob_mask)
                    head_log_prob *= log_prob_mask
                joint_action_log_prob += head_log_prob
                head_log_probs_filtered.append((log_prob_mask==0).float().detach())

                entropy_head = log_prob_mask * (head_distribution.entropy().view(-1, 1))
                entropy += entropy_head.mean()
            else:
                if actions is None:
                    action_inp = None
                    prev_head_acs = action_outputs
                else:
                    action_inp = actions[i]
                    prev_head_acs = actions
                head_output, action_output, head_log_prob, head_entropy = \
                    head(main_head_inputs, head_mask, custom_inputs, self.log_prob_masks[i], head_log_probs_filtered,
                         prev_head_acs, actions=action_inp, deterministic=deterministic)
                head_outputs.append(head_output)
                action_outputs.append(action_output)
                joint_action_log_prob += head_log_prob
                entropy += head_entropy
                head_log_probs_filtered.append((head_log_prob==0).float().detach())

        return action_outputs, joint_action_log_prob, entropy


class ActionHead(nn.Module):
    def __init__(self, main_input_dim, output_dim, custom_inputs={}, type="categorical", mlp_size=None,
                 returns_distribution=True, id=None):
        super(ActionHead, self).__init__()
        self.input_dim = main_input_dim
        for name, size in custom_inputs.items():
            self.input_dim += size
        self.output_dim = output_dim
        self.type = type
        self.mlp_size = mlp_size
        self.returns_distribution = returns_distribution
        self.custom_inputs = custom_inputs
        self.id = id
        if mlp_size is None:
            dist_input_size = self.input_dim
        else:
            self.mlp = nn.Linear(self.input_dim, mlp_size)
            dist_input_size = mlp_size
        if type == "categorical":
            self.distribution = Categorical(num_inputs=dist_input_size, num_outputs=output_dim)
        elif type == "normal":
            self.distribution = DiagGaussian(num_inputs=dist_input_size, num_outputs=output_dim)
        else:
            raise NotImplementedError

    def forward(self, main_input, mask, custom_inputs):
        if custom_inputs is not None and len(self.custom_inputs) > 0:
            input_custom = torch.cat([custom_inputs[key] for key in self.custom_inputs.keys()], dim=-1)
            input_full = torch.cat([main_input, input_custom], dim=-1)
        else:
            input_full = main_input
        if self.mlp_size is not None:
            head_input = self.mlp(input_full)
        else:
            head_input = input_full

        if self.type == "normal":
            return self.distribution(head_input)
        else:
            return self.distribution(head_input, mask)



class RecurrentResourceActionHead(nn.Module):
    """specific to the Settler's environment. For trading resources. Assume zeroth entry is stop."""
    def __init__(self, main_input_dim, available_resources_dim, max_count=4, custom_inputs={}, mlp_size=None,
                 id=None, mask_based_on_curr_res=True):
        super(RecurrentResourceActionHead, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.main_input_dim = main_input_dim
        self.available_resources_dim = available_resources_dim
        self.max_count = max_count
        self.custom_inputs = custom_inputs
        self.mlp_size = mlp_size
        self.returns_distribution = False
        self.mask_based_on_curr_res = mask_based_on_curr_res
        self.input_dim = main_input_dim + available_resources_dim
        self.id = id
        for name, size in custom_inputs.items():
            self.input_dim += size
        if mlp_size is None:
            dist_input_size = self.input_dim
        else:
            self.mlp = nn.Linear(self.input_dim, mlp_size)
            dist_input_size = mlp_size

        self.distribution = Categorical(num_inputs=dist_input_size, num_outputs=available_resources_dim)

    def forward(self, main_inputs, head_mask, custom_inputs, log_prob_masks, filtered_heads, prev_head_action_outs,
                actions=None, deterministic=False, count=None):
        """head mask here is just a placeholder. mask is based on available resources"""
        actions_out = []
        output = torch.zeros(main_inputs.size(0), self.available_resources_dim, dtype=torch.float32,
                             device=self.dummy_param.device)
        log_prob_sum = 0
        entropy_sum = 0
        current_resources = custom_inputs["current_resources"]  # in one-hot encoded form where first entry is 0 and represents stop/no res
        if self.mask_based_on_curr_res:
            mask = (current_resources > 0).float()
        else:
            mask = torch.ones_like(current_resources, dtype=torch.float32, device=self.dummy_param.device)
        res_sum = torch.sum(current_resources, dim=-1)
        zero_res_mask = (res_sum == 0)
        mask[:, 0] = 0.0
        mask[zero_res_mask, 0] = 1.0 #allow no res as first res if sum of current res = 0 (logits will be masked out anyway but leads to error o/w)
        if count is None:
            count = self.max_count
        for i in range(count):
            input = torch.cat((main_inputs, output), dim=-1)
            if self.mlp_size is not None:
                input = self.mlp(input)
            distribution = self.distribution(input, mask)

            if deterministic:
                action = distribution.mode()
            else:
                action = distribution.sample()

            one_hot_action = torch.zeros(main_inputs.size(0), self.available_resources_dim, dtype=torch.float32,
                                         device=self.dummy_param.device)
            if actions is None:
                one_hot_action.scatter_(-1, action, 1.0)
                log_prob = distribution.log_probs(action)
            else:
                one_hot_action.scatter_(-1, actions[:, i].view(-1, 1), 1.0)
                log_prob = distribution.log_probs(actions[:, i])
            output += one_hot_action
            current_resources = torch.clamp(current_resources - one_hot_action, 0, math.inf)

            if self.mask_based_on_curr_res:
                mask = (current_resources > 0).float()
            else:
                mask = torch.ones_like(current_resources, dtype=torch.float32, device=self.dummy_param.device)
            mask[:, 0] = 1.0

            entropy = distribution.entropy().view(-1, 1)

            if i > 0:
                if actions is not None:
                    log_prob_mask = (actions[:, i-1] > 0).float().view(-1, 1)
                else:
                    log_prob_mask = (actions_out[-1][:, 0] > 0).float().view(-1, 1)
                log_prob *= log_prob_mask
                entropy *= log_prob_mask

            log_prob_sum += log_prob
            entropy_sum += entropy

            actions_out.append(action)

            output[:, 0] = 0.0

        #now apply log_prob_mask from other heads
        log_prob_mask = 1
        for prev_head_ind, head_type_mask in log_prob_masks.items():
            filter = filtered_heads[prev_head_ind]
            head_prob_mask = head_type_mask[prev_head_action_outs[prev_head_ind].squeeze()].view(-1, 1)
            log_prob_mask *= ((1-filter) * head_prob_mask)
        log_prob_sum *= log_prob_mask
        entropy_sum *= log_prob_mask

        return output, actions_out, log_prob_sum, entropy_sum.mean()