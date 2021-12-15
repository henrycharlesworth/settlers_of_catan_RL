import random
import torch
import copy

from RL.models.build_agent_model import build_agent_model
from env.wrapper import EnvWrapper

from game.enums import PlayerId

class GamesAndPoliciesManager(object):
    def __init__(self, num_envs=1, num_steps=20, policy_kwargs={}):
        self.num_envs = num_envs
        self.num_steps = num_steps

        self.policies = [build_agent_model(**policy_kwargs) for _ in range(4)]
        self.envs = [EnvWrapper() for _ in range(num_envs)]
        self.device = self.policies[0].dummy_param.device

        self.initialise()

    def initialise(self):
        self.policy_maps = []
        self.active_player_ids = []
        for i in range(self.num_envs):
            order = [PlayerId.Blue, PlayerId.Red, PlayerId.Orange, PlayerId.White]
            random.shuffle(order)
            self.active_player_ids.append(order[0]) #active player controlled by policy[0]
            policy_map = {}
            for j in range(4):
                policy_map[order[j]] = self.policies[j]
            self.policy_maps.append(policy_map)

        self.reset()

    def reset(self):
        self.observations = [[] for _ in range(self.num_envs)]
        self.action_masks = [[] for _ in range(self.num_envs)]
        self.active_hidden_states = [[] for _ in range(self.num_envs)]
        self.actions = [[] for _ in range(self.num_envs)]
        self.action_log_probs = [[] for _ in range(self.num_envs)]
        self.rewards = [[] for _ in range(self.num_envs)]
        self.terminal_masks = [[] for _ in range(self.num_envs)]

        self.current_hidden_states = [{} for _ in range(self.num_envs)]
        self.current_observations = [{} for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            obs = self.policies[0].obs_to_torch(self.envs[i].reset())
            self.terminal_masks[i].append(1.0)  # IS THIS RIGHT??
            players_go = self._get_players_turn(self.envs[i])
            self.current_observations[i][players_go] = obs
            if players_go == self.active_player_ids[i]:
                self.observations[i].append(obs)
                self.active_hidden_states[i].append((torch.zeros(1, self.policies[0].lstm_size, device=self.device),
                                                     torch.zeros(1, self.policies[0].lstm_size, device=self.device)))

            for player_id in [PlayerId.Blue, PlayerId.Red, PlayerId.Orange, PlayerId.White]:
                self.current_hidden_states[i][player_id] = (torch.zeros(1, self.policies[0].lstm_size, device=self.device),
                                                       torch.zeros(1, self.policies[0].lstm_size, device=self.device))

    def shuffle_policies(self, env_num):
        order = [PlayerId.Blue, PlayerId.Red, PlayerId.Orange, PlayerId.White]
        random.shuffle(order)
        policy_map = []
        for j in range(4):
            policy_map[order[j]] = self.policies[j]
        self.policy_maps[env_num] = policy_map

    def gather_rollouts(self):
        for i in range(4):
            self.policies[i].eval()
        with torch.no_grad():
            for env_num in range(self.num_envs):
                terminal_mask = torch.tensor(self.terminal_masks[env_num][0],
                                             dtype=torch.float32, device=self.device).view(1,1)
                rewards = {player_id: 0 for player_id in [PlayerId.White, PlayerId.Blue, PlayerId.Red, PlayerId.Orange]}
                done_since_prev_turn = [False for _ in range(self.num_envs)]
                while len(self.observations[env_num]) < self.num_steps + 1:
                    players_go = self._get_players_turn(self.envs[env_num])

                    obs = self.current_observations[env_num][players_go]
                    hidden_states = self.current_hidden_states[env_num][players_go]
                    action_masks = self.policies[0].act_masks_to_torch(self.envs[env_num].get_action_masks())

                    _, actions, action_log_probs, hidden_states = self.policy_maps[env_num][players_go].act(
                        obs, hidden_states, terminal_mask, action_masks
                    )

                    self.current_hidden_states[env_num][players_go] = hidden_states

                    obs, reward, done, _ = self.envs[env_num].step(self.policies[0].torch_act_to_np(actions))
                    obs = self.policies[0].obs_to_torch(obs)

                    for player_id in [PlayerId.White, PlayerId.Blue, PlayerId.Red, PlayerId.Orange]:
                        rewards[player_id] += reward[player_id]

                    terminal_mask = 1.0 - torch.tensor(done, dtype=torch.float32, device=self.device).view(1,1)

                    n_players_go = self._get_players_turn(self.envs[env_num])

                    reward_updated = False
                    if players_go == self.active_player_ids[env_num]:
                        self.actions[env_num].append(actions)
                        self.action_log_probs[env_num].append(action_log_probs)
                        self.action_masks[env_num].append(action_masks)
                    if n_players_go == self.active_player_ids[env_num] and len(self.actions[env_num]) > 0:
                        if done_since_prev_turn[env_num] == False:
                            self.rewards[env_num].append(rewards[self.active_player_ids[env_num]])
                            rewards[self.active_player_ids[env_num]] = 0.0
                            reward_updated = True

                    if done:
                        obs = self.policies[0].obs_to_torch(self.envs[env_num].reset())
                        self.terminal_masks[env_num].append(1.0 - done)
                        done_since_prev_turn[env_num] = False
                        n_players_go = self._get_players_turn(self.envs[env_num])
                        if reward_updated == False:
                            self.rewards[env_num].append(rewards[self.active_player_ids[env_num]])

                        for player_id in [PlayerId.White, PlayerId.Blue, PlayerId.Red, PlayerId.Orange]:
                            rewards[player_id] = 0.0
                            self.current_hidden_states[env_num][player_id] = (
                                                torch.zeros(1, self.policies[0].lstm_size, device=self.device),
                                                torch.zeros(1, self.policies[0].lstm_size, device=self.device))

                    self.current_observations[env_num][n_players_go] = obs

                    if n_players_go == self.active_player_ids[env_num]:
                        if done == False and done_since_prev_turn[env_num]== False:
                            self.terminal_masks[env_num].append(1.0-done)
                        done_since_prev_turn[env_num] = False
                        self.observations[env_num].append(obs)
                        self.active_hidden_states[env_num].append(self.current_hidden_states[env_num][n_players_go])
                    else:
                        if done:
                            done_since_prev_turn[env_num] = True
        return copy.deepcopy((
            self.observations, self.active_hidden_states, self.rewards, self.actions, self.action_masks,
            self.action_log_probs, self.terminal_masks
        ))

    def _after_rollouts(self):
        for env_num in range(self.num_envs):
            self.observations[env_num] = [self.observations[env_num][-1]]
            self.active_hidden_states[env_num] = [self.active_hidden_states[env_num][-1]]
            self.terminal_masks[env_num] = [self.terminal_masks[env_num][-1]]
            self.actions[env_num] = []
            self.action_masks[env_num] = []
            self.action_log_probs[env_num] = []
            self.rewards[env_num] = []

    def _get_players_turn(self, env):
        if env.game.players_need_to_discard:
            player_id = env.game.players_to_discard[0]
        elif env.game.must_respond_to_trade:
            player_id = env.game.proposed_trade["target_player"]
        else:
            player_id = env.game.players_go
        return player_id

    def _update_policy(self, policy_dict, policy_id=0):
        self.policies[policy_id].load_state_dict(policy_dict)

    def _update_annealing_factor(self, annealing_factor):
        for env in self.envs:
            env.reward_annealing_factor = annealing_factor

def make_game_manager(num_envs, num_steps):
    def _thunk():
        manager = GamesAndPoliciesManager(num_envs=num_envs, num_steps=num_steps)
        return manager
    return _thunk