import random
import torch
from collections import defaultdict
import numpy as np

from RL.models.build_agent_model import build_agent_model
from env.wrapper import EnvWrapper

from game.enums import PlayerId


class EvaluationManager(object):
    def __init__(self, policies=None, policy_kwargs={}, detailed_logging=False):
        self.detailed_logging = detailed_logging

        if policies is not None:
            self.policies = policies
        else:
            self.policies = [build_agent_model(**policy_kwargs) for _ in range(4)]
        self.env = EnvWrapper()
        self.device = self.policies[0].dummy_param.device

    def reset(self):
        self.current_hidden_states = {}
        self.current_observations = {}

        for player_id in [PlayerId.Blue, PlayerId.Red, PlayerId.Orange, PlayerId.White]:
            self.current_hidden_states[player_id] = (torch.zeros(1, self.policies[-1].lstm_size, device=self.device),
                                                       torch.zeros(1, self.policies[-1].lstm_size, device=self.device))
        self.order = [PlayerId.Blue, PlayerId.Red, PlayerId.Orange, PlayerId.White]
        random.shuffle(self.order)
        self.policy_map = {}
        for i, player_id in enumerate(self.order):
            self.policy_map[player_id] = i
            self.policies[i].player_id = player_id

            if hasattr(self.policies[i], "initialise_policy"):
                self.policies[i].initialise_policy()

        obs = self.policies[-1].obs_to_torch(self.env.reset())
        players_go = self._get_players_turn(self.env)
        self.current_observations[players_go] = obs

    def run_evaluation_game(self):
        for i in range(4):
            self.policies[i].eval()
        self.reset()
        terminal_mask = torch.ones(1, 1, device=self.device)
        done = False
        total_game_steps = 0
        policy_decisions = 0
        DRAW = False

        entropies = []
        action_types = defaultdict(lambda: 0)
        type_prob_tuples = []
        values = []
        detailed_action_outputs = []

        while done == False:
            players_go = self._get_players_turn(self.env)
            with torch.no_grad():
                obs = self.current_observations[players_go]
                hidden_states = self.current_hidden_states[players_go]
                action_masks = self.policies[-1].act_masks_to_torch(self.env.get_action_masks())

                if self.policies[self.policy_map[players_go]].policy_type == "neural_network":
                    if self.detailed_logging:
                        value, actions, action_log_probs, hidden_states, entropy, detailed_action_out = self.policies[
                            self.policy_map[players_go]].act(
                            obs, hidden_states, terminal_mask, action_masks, deterministic=False,
                            return_entropy=True, log_specific_action_output=True
                        )
                    else:
                        value, actions, action_log_probs, hidden_states, entropy = self.policies[self.policy_map[players_go]].act(
                            obs, hidden_states, terminal_mask, action_masks, deterministic=False,
                            return_entropy=True
                        )
                    entropy = entropy.detach().cpu().data.numpy()
                    value = value.detach().squeeze().data.numpy()
                    action_log_probs = action_log_probs.detach().squeeze().data.numpy()
                    actions = self.policies[-1].torch_act_to_np(actions)
                elif self.policies[self.policy_map[players_go]].policy_type == "forward_search":
                    curr_state = self.env.save_state()
                    placing_initial_settlement = False
                    if self.env.game.initial_settlements_placed[players_go] == 0:
                        placing_initial_settlement = True
                    elif self.env.game.initial_settlements_placed[players_go] == 1 and self.env.game.initial_roads_placed[players_go] == 1:
                        placing_initial_settlement = True
                    actions, hidden_states = self.policies[self.policy_map[players_go]].act(
                        obs, self.current_hidden_states, curr_state, action_masks, decision_no=policy_decisions,
                        initial_settlement = placing_initial_settlement
                    )
                    entropy = 0.0
                    action_log_probs = 0.0
                    value = 0.0

                if self.policy_map[players_go] == 0:
                    policy_decisions += 1
                    entropies.append(entropy)
                    action_type = int(actions[0].ravel())
                    action_types[action_type] += 1
                    type_prob_tuples.append((action_type, action_log_probs))
                    values.append(value)
                    if self.detailed_logging:
                        detailed_action_outputs.append(detailed_action_out)

                self.current_hidden_states[players_go] = hidden_states

                obs, reward, done, _ = self.env.step(actions)
                obs = self.policies[-1].obs_to_torch(obs)

                n_players_go = self._get_players_turn(self.env)

                self.current_observations[n_players_go] = obs

                total_game_steps += 1

                if total_game_steps > 2500:
                    DRAW = True
                    done = True
                    # print("done")

        if DRAW:
            winner = -1
        else:
            winner = self.order.index(self.env.winner.id)

        victory_points = self.env.curr_vps[self.order[0]]

        if self.detailed_logging:
            return winner, victory_points, total_game_steps, policy_decisions, np.mean(entropies), action_types, type_prob_tuples, np.mean(values), detailed_action_outputs
        else:
            return winner, victory_points, total_game_steps, policy_decisions, np.mean(entropies), action_types, type_prob_tuples, np.mean(values)

    def _get_players_turn(self, env):
        if env.game.players_need_to_discard:
            player_id = env.game.players_to_discard[0]
        elif env.game.must_respond_to_trade:
            player_id = env.game.proposed_trade["target_player"]
        else:
            player_id = env.game.players_go
        return player_id

    def _update_policies(self, state_dicts):
        for i in range(4):
            self.policies[i].load_state_dict(state_dicts[i])


def make_evaluation_manager():
    def _thunk():
        manager = EvaluationManager()
        return manager
    return _thunk