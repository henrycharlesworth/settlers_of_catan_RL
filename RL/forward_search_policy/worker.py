import torch
import numpy as np
import copy
import multiprocessing as mp

from RL.models.build_agent_model import build_agent_model
from env.wrapper import EnvWrapper

def worker(remote: mp.connection.Connection, parent_remote: mp.connection.Connection, shared_queue: mp.Queue,
           worker_id=0) -> None:
    parent_remote.close()

    torch.set_num_threads(1)

    env = EnvWrapper(dense_reward=True)
    env.reset()
    policy = build_agent_model(device="cpu")

    gamma = None

    initial_state_to_load = None
    starting_hidden_states = None
    starting_observation = None

    first_actions_under_consideration = None
    player_hidden_states_after_first_ac = None
    controlling_player_id = None

    while True:
        cmd, data = remote.recv()
        if cmd == "initialise_policy":
            state_dict, controlling_player_id, gamma = data.var
            policy.load_state_dict(state_dict)
            policy.eval()
            remote.send(True)
        elif cmd == "set_start_states":
            initial_state_to_load, starting_hidden_states, starting_observation = data.var
            remote.send(True)
        elif cmd == "set_initial_actions":
            first_actions_under_consideration, player_hidden_states_after_first_ac = data.var
            remote.send(True)
        elif cmd == "run_simulation":
            remote.send(True) #let main process know it's started simulation.
            env.reset()
            env.restore_state(initial_state_to_load)
            env.game.randomise_uncertainty(controlling_player_id)

            init_ac_id = data[0]
            max_depth = data[1]

            pred_val = run_simulation_forward(env, policy, player_id=controlling_player_id,
                                              init_action=first_actions_under_consideration[init_ac_id],
                                              init_player_hs=player_hidden_states_after_first_ac[init_ac_id],
                                              curr_hidden_states = copy.deepcopy(starting_hidden_states),
                                              curr_obs = copy.deepcopy(starting_observation),
                                              max_depth=max_depth, gamma=gamma)

            shared_queue.put((pred_val, init_ac_id, worker_id))


def run_simulation_forward(env, policy, player_id, init_action, init_player_hs, curr_hidden_states, curr_obs,
                           max_depth=20, gamma=0.999):
    current_observations = {player_id: curr_obs}
    actual_rewards = []
    values = []

    agent_actions = 0
    game_finished = False

    #first step
    next_obs, reward, done, _ = env.step(init_action)
    next_obs = policy.obs_to_torch(next_obs)
    agent_actions += 1
    curr_hidden_states[player_id] = init_player_hs
    actual_rewards.append(reward[player_id])
    if done:
        return actual_rewards[0]
    n_players_go = get_players_turn(env)
    current_observations[n_players_go] = next_obs

    while agent_actions < max_depth:
        players_go = get_players_turn(env)
        obs = current_observations[players_go]
        hidden_state = curr_hidden_states[players_go]
        action_masks = policy.act_masks_to_torch(env.get_action_masks())
        terminal_mask = torch.ones(1,1)

        value, action, _, next_hidden_state = policy.act(
            obs, hidden_state, terminal_mask, action_masks
        )

        if policy.use_value_normalisation:
            value = policy.value_normaliser.denormalise(value)

        curr_hidden_states[players_go] = copy.copy(next_hidden_state)

        next_obs, reward, done, _ = env.step(policy.torch_act_to_np(action))

        if players_go == player_id:
            agent_actions += 1
            actual_rewards.append(reward[player_id])
            values.append(value.squeeze().cpu().data.numpy())
        elif done:
            actual_rewards.append(reward[player_id])

        n_players_go = get_players_turn(env)
        current_observations[n_players_go] = policy.obs_to_torch(next_obs)

        if done:
            game_finished = True
            break

    final_val_est = gae(values, actual_rewards, gamma=gamma, done=game_finished)
    return final_val_est


def gae(values, rewards, gamma, done):
    """generalised advantage estimation"""
    gae_lambda = 0.95

    if len(values) <= 1:
        return rewards[0] + gamma * rewards[1]

    first_rew = rewards[0]
    rewards_for_gae = rewards[1:-1]
    values_for_gae = values[:-1]
    if done:
        final_value = 0.0
    else:
        final_value = values[-1]
    num_steps = len(values_for_gae) - 1
    values.append(final_value)
    values = np.array(values)

    gae = 0.0
    for step in reversed(range(num_steps)):
        delta = rewards_for_gae[step] + gamma * values_for_gae[step + 1] - values_for_gae[step]
        if step == (num_steps - 1) and done:
            gae = delta
        else:
            gae = delta + gamma * gae_lambda * gae
    returns_from_state = gae + values_for_gae[0]
    return first_rew + gamma * returns_from_state


def get_players_turn(env):
    if env.game.must_respond_to_trade:
        player_id = env.game.proposed_trade["target_player"]
    else:
        player_id = env.game.players_go
    return player_id