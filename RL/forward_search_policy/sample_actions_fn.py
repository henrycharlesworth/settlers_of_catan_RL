import numpy as np
import torch
import copy
import random

max_prop_trade_actions = 3

action_type_priorities = {
    1: ["settlement", "city", "move_robber", "steal", "discard"],
    2: ["road"],
    3: ["play_dev"],
    4: ["exchange_res", "prop_trade"]
}

type_to_ind = {
    "settlement": 0,
    "road": 1,
    "city": 2,
    "buy_dev": 3,
    "play_dev": 4,
    "exchange_res": 5,
    "prop_trade": 6,
    "respond_trade": 7,
    "move_robber": 8,
    "roll_dice": 9,
    "end_turn": 10,
    "steal": 11,
    "discard": 12
}

def update_action_masks(action, action_masks):
    if action[0] == 0:
        if sum(action_masks[1][0]) > 1:
            action_masks[1][0][action[1]] = 0
    elif action[0] == 1:
        if sum(action_masks[2]) > 1:
            action_masks[2][action[2]] = 0
    elif action[0] == 2:
        if sum(action_masks[1][1]) > 1:
            action_masks[1][1][action[1]] = 0
    elif action[0] == 4:
        if sum(action_masks[4]) > 1:
            action_masks[4][action[4]] = 0
    elif action[0] == 8:
        if sum(action_masks[3]) > 0:
            action_masks[3][action[3]] = 0
    elif action[0] == 11:
        if sum(action_masks[6][1]) > 0:
            action_masks[6][1][action[6]] = 0
    elif action[0] == 12:
        if sum(action_masks[11]) > 1:
            action_masks[11][action[11]] = 0
    return action_masks

def default_sample_actions(obs, hidden_state, action_masks, policy, max_actions,
                           dont_propose_devcards=False, dont_propose_trades=False,
                           consider_all_initial_settlements=False, initial_settlement_phase=False):
    effective_actions_available = 0
    actions_available_type = {}

    action_masks_torch = copy.copy(action_masks)

    for i in range(len(action_masks)):
        action_masks[i] = action_masks[i].squeeze().cpu().data.numpy()

    type_masks = action_masks[0]

    terminal_mask = torch.ones(1, 1)

    proposed_actions = []
    hidden_states_after = []

    exchanges_proposed = []

    actions_sampled = 0
    trades_proposed = 0

    if type_masks[0] == 1: #place settlement
        actions_available_type["settlement"] = np.sum(action_masks[1][0]) - 1
        effective_actions_available += actions_available_type["settlement"]

        with torch.no_grad():
            _, action, _, next_hs = policy.act(obs, hidden_state, terminal_mask, action_masks_torch, deterministic=False,
                                                          condition_on_action_type=0)
        action = policy.torch_act_to_np(action)
        assert action[0] == 0

        proposed_actions.append(action)
        hidden_states_after.append(copy.deepcopy(next_hs))
        actions_sampled += 1

        action_masks = update_action_masks(action, action_masks)
        action_masks_torch = policy.act_masks_to_torch(copy.copy(action_masks))

    if type_masks[1] == 1: #build road
        actions_available_type["road"] = np.sum(action_masks[2]) - 1
        effective_actions_available += actions_available_type["road"]

        with torch.no_grad():
            _, action, _, next_hs = policy.act(obs, hidden_state, terminal_mask, action_masks_torch,
                                                          deterministic=False, condition_on_action_type=1)
        action = policy.torch_act_to_np(action)
        assert action[0] == 1

        proposed_actions.append(action)
        hidden_states_after.append(copy.deepcopy(next_hs))
        actions_sampled += 1

        action_masks = update_action_masks(action, action_masks)
        action_masks_torch = policy.act_masks_to_torch(copy.copy(action_masks))

    if type_masks[2] == 1: #upgrade to city
        actions_available_type["city"] = np.sum(action_masks[1][1]) - 1
        effective_actions_available += actions_available_type["city"]

        with torch.no_grad():
            _, action, _, next_hs = policy.act(obs, hidden_state, terminal_mask, action_masks_torch,
                                                    deterministic=False, condition_on_action_type=2)
        action = policy.torch_act_to_np(action)
        assert action[0] == 2

        proposed_actions.append(action)
        hidden_states_after.append(copy.deepcopy(next_hs))
        actions_sampled += 1

        action_masks = update_action_masks(action, action_masks)
        action_masks_torch = policy.act_masks_to_torch(copy.copy(action_masks))

    if type_masks[3] == 1: #buy dev card
        if dont_propose_devcards == False:
            effective_actions_available += 0

            with torch.no_grad():
                _, action, _, next_hs = policy.act(obs, hidden_state, terminal_mask, action_masks_torch,
                                                        deterministic=False, condition_on_action_type=3)
            action = policy.torch_act_to_np(action)
            assert action[0] == 3

            proposed_actions.append(action)
            hidden_states_after.append(copy.deepcopy(next_hs))
            actions_sampled += 1

    if type_masks[4] == 1: #play dev card
        if dont_propose_devcards == False:
            actions_available_type["play_dev"] = np.sum(action_masks[4]) - 1
            effective_actions_available += actions_available_type["play_dev"]

            with torch.no_grad():
                _, action, _, next_hs = policy.act(obs, hidden_state, terminal_mask, action_masks_torch,
                                                   deterministic=False, condition_on_action_type=4)
            action = policy.torch_act_to_np(action)
            assert action[0] == 4

        proposed_actions.append(action)
        hidden_states_after.append(copy.deepcopy(next_hs))
        actions_sampled += 1

        action_masks = update_action_masks(action, action_masks)
        action_masks_torch = policy.act_masks_to_torch(copy.copy(action_masks))

    if type_masks[5] == 1: #exchange res
        actions_available_type["exchange_res"] = np.sum(action_masks[10]) * np.sum(action_masks[9][0]) - 1
        effective_actions_available += actions_available_type["exchange_res"]

        with torch.no_grad():
            _, action, _, next_hs = policy.act(obs, hidden_state, terminal_mask, action_masks_torch,
                                               deterministic=False, condition_on_action_type=5)
        action = policy.torch_act_to_np(action)
        assert action[0] == 5

        proposed_actions.append(action)
        hidden_states_after.append(copy.deepcopy(next_hs))
        actions_sampled += 1

        exchanges_proposed.append(str(action[9]) + "_" + str(action[10]))

    if type_masks[6] == 1: #prop trade
        if dont_propose_trades == False:
            actions_available_type["prop_trade"] = max_prop_trade_actions - 1
            effective_actions_available += actions_available_type["prop_trade"]

            with torch.no_grad():
                _, action, _, next_hs = policy.act(obs, hidden_state, terminal_mask, action_masks_torch,
                                                   deterministic=False, condition_on_action_type=6)
            action = policy.torch_act_to_np(action)
            assert action[0] == 6

            proposed_actions.append(action)
            hidden_states_after.append(copy.deepcopy(next_hs))
            actions_sampled += 1
            trades_proposed += 1

    if type_masks[7] == 1: #respond to trade
        effective_actions_available += np.sum(action_masks[5]) - 1

        with torch.no_grad():
            _, action, _, next_hs = policy.act(obs, hidden_state, terminal_mask, action_masks_torch,
                                               deterministic=False, condition_on_action_type=7)
        action = policy.torch_act_to_np(action)
        assert action[0] == 7

        proposed_actions.append(action)
        hidden_states_after.append(copy.deepcopy(next_hs))
        actions_sampled += 1

    if type_masks[8] == 1: #move robber
        actions_available_type["move_robber"] = np.sum(action_masks[3]) - 1
        effective_actions_available += actions_available_type["move_robber"]

        with torch.no_grad():
            _, action, _, next_hs = policy.act(obs, hidden_state, terminal_mask, action_masks_torch,
                                               deterministic=False, condition_on_action_type=8)
        action = policy.torch_act_to_np(action)
        assert action[0] == 8

        proposed_actions.append(action)
        hidden_states_after.append(copy.deepcopy(next_hs))
        actions_sampled += 1

        action_masks = update_action_masks(action, action_masks)
        action_masks_torch = policy.act_masks_to_torch(copy.copy(action_masks))

    if type_masks[9] == 1: #roll dice
        effective_actions_available += 0

        with torch.no_grad():
            _, action, _, next_hs = policy.act(obs, hidden_state, terminal_mask, action_masks_torch,
                                               deterministic=False, condition_on_action_type=9)
        action = policy.torch_act_to_np(action)
        assert action[0] == 9

        proposed_actions.append(action)
        hidden_states_after.append(copy.deepcopy(next_hs))
        actions_sampled += 1

    if type_masks[10] == 1: #end turn
        effective_actions_available += 0

        with torch.no_grad():
            _, action, _, next_hs = policy.act(obs, hidden_state, terminal_mask, action_masks_torch,
                                               deterministic=False, condition_on_action_type=10)
        action = policy.torch_act_to_np(action)
        assert action[0] == 10

        proposed_actions.append(action)
        hidden_states_after.append(copy.deepcopy(next_hs))
        actions_sampled += 1

    if type_masks[11] == 1: #steal
        actions_available_type["steal"] = np.sum(action_masks[6][1]) - 1
        effective_actions_available += actions_available_type["steal"]

        with torch.no_grad():
            _, action, _, next_hs = policy.act(obs, hidden_state, terminal_mask, action_masks_torch,
                                               deterministic=False, condition_on_action_type=11)
        action = policy.torch_act_to_np(action)
        assert action[0] == 11

        proposed_actions.append(action)
        hidden_states_after.append(copy.deepcopy(next_hs))
        actions_sampled += 1

        action_masks = update_action_masks(action, action_masks)
        action_masks_torch = policy.act_masks_to_torch(copy.copy(action_masks))

    if type_masks[12] == 1: #discard
        actions_available_type["discard"] = np.sum(action_masks[11]) - 1
        effective_actions_available += actions_available_type["discard"]

        with torch.no_grad():
            _, action, _, next_hs = policy.act(obs, hidden_state, terminal_mask, action_masks_torch,
                                               deterministic=False, condition_on_action_type=12)
        action = policy.torch_act_to_np(action)
        assert action[0] == 12

        proposed_actions.append(action)
        hidden_states_after.append(copy.deepcopy(next_hs))
        actions_sampled += 1

        action_masks = update_action_masks(action, action_masks)
        action_masks_torch = policy.act_masks_to_torch(copy.copy(action_masks))

    if initial_settlement_phase:
        if consider_all_initial_settlements:
            num_actions_to_sample = int(actions_available_type["settlement"])
        else:
            num_actions_to_sample = int(min(max_actions-actions_sampled, effective_actions_available))
    else:
        num_actions_to_sample = int(min(max_actions-actions_sampled, effective_actions_available))

    num_priorities = len(action_type_priorities.keys())

    for i in range(num_actions_to_sample):
        ac_type = None
        for j in range(num_priorities):
            avail_types = [ac_type for ac_type in action_type_priorities[j+1] if actions_available_type.get(ac_type, 0) > 0]
            if trades_proposed >= max_prop_trade_actions and "prop_trade" in avail_types:
                avail_types.remove("prop_trade")
            if len(avail_types) > 0:
                ac_type = random.choice(avail_types)
                actions_available_type[ac_type] -= 1
                break
        if ac_type is None:
            break #something gone wrong - but just return what we have.

        with torch.no_grad():
            _, policy_action, _, next_hs = policy.act(obs, hidden_state, terminal_mask, action_masks_torch,
                                                      deterministic=False, condition_on_action_type=type_to_ind[ac_type])
        policy_action = policy.torch_act_to_np(policy_action)

        hidden_states_after.append(next_hs)
        if ac_type == "prop_trade":
            proposed_actions.append(policy_action)
            trades_proposed += 1
            continue
        elif ac_type == "exchange_res":
            prop_exchange = str(policy_action[9]) + "_" + str(policy_action[10])
            while prop_exchange not in exchanges_proposed:
                policy_action[9] = random.choice(np.where(action_masks[9][0])[0])
                policy_action[10] = random.choice(np.where(action_masks[10])[0])
                prop_exchange = str(policy_action[9]) + "_" + str(policy_action[10])
            proposed_actions.append(policy_action)
            exchanges_proposed.append(prop_exchange)
        else:
            proposed_actions.append(policy_action)
            action_masks = update_action_masks(policy_action, action_masks)
            action_masks_torch = policy.act_masks_to_torch(copy.copy(action_masks))

    return proposed_actions, hidden_states_after