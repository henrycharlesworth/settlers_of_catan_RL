"""probably should have integrated this with game_manager, but oh well. Slightly clearer like this anyway I think."""

import random
import torch

from RL.models.build_agent_model import build_agent_model
from env.wrapper import EnvWrapper

from game.enums import PlayerId


class EvaluationManager(object):
    def __init__(self, policy_kwargs={}):

        self.policies = [build_agent_model(**policy_kwargs) for _ in range(4)]
        self.env = EnvWrapper()
        self.device = self.policies[0].dummy_param.device

    def reset(self):
        self.current_hidden_states = {}
        self.current_observations = {}

        for player_id in [PlayerId.Blue, PlayerId.Red, PlayerId.Orange, PlayerId.White]:
            self.current_hidden_states[player_id] = (torch.zeros(1, self.policies[0].lstm_size, device=self.device),
                                                       torch.zeros(1, self.policies[0].lstm_size, device=self.device))
        self.order = [PlayerId.Blue, PlayerId.Red, PlayerId.Orange, PlayerId.White]
        random.shuffle(self.order)
        self.policy_map = {}
        for i, player_id in enumerate(self.order):
            self.policy_map[player_id] = i

        obs = self.policies[0].obs_to_torch(self.env.reset())
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

        while done == False:
            with torch.no_grad():
                players_go = self._get_players_turn(self.env)

                obs = self.current_observations[players_go]
                hidden_states = self.current_hidden_states[players_go]
                action_masks = self.policies[0].act_masks_to_torch(self.env.get_action_masks())
                _, actions, _, hidden_states = self.policies[self.policy_map[players_go]].act(
                    obs, hidden_states, terminal_mask, action_masks, deterministic=False
                )

                if self.policy_map[players_go] == 0:
                    policy_decisions += 1

                self.current_hidden_states[players_go] = hidden_states

                obs, reward, done, _ = self.env.step(self.policies[0].torch_act_to_np(actions))
                obs = self.policies[0].obs_to_torch(obs)

                n_players_go = self._get_players_turn(self.env)

                self.current_observations[n_players_go] = obs

                total_game_steps += 1

        winner = self.order.index(self.env.winner.id)

        # print(winner)

        victory_points = self.env.curr_vps[self.order[0]]
        return winner, victory_points, total_game_steps, policy_decisions

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