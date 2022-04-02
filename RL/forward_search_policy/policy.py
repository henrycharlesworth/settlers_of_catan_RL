import time
import torch
import math
import numpy as np
import multiprocessing as mp
from collections import deque

from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper

from RL.models.build_agent_model import build_agent_model
from RL.forward_search_policy.worker import worker
from RL.forward_search_policy.utils import MovingAvgCalculator

from game.enums import PlayerId


class ForwardSearchPolicy(object):
    def __init__(self, base_policy_state_dict, sample_actions_fn, max_init_actions=10, max_depth=20,
                 max_thinking_time=10, gamma=0.999, num_subprocesses=11, subprocess_start_method=None,
                 player_id=None, zero_opponent_hidden_states=True, consider_all_moves_for_opening_placement=False,
                 dont_propose_devcards=False, dont_propose_trades=False, lstm_size=256):
        self.base_policy = build_agent_model(device="cpu")
        self.base_policy.load_state_dict(base_policy_state_dict)
        self.player_id = player_id
        self.dummy_param = torch.empty(size=(1,), device="cpu")

        self.policy_type = "forward_search"
        self.lstm_size = lstm_size

        self.standard_obs_keys = ["proposed_trade", "current_resources", "current_player_main", "next_player_main",
                                  "next_next_player_main", "next_next_next_player_main"]
        self.list_int_obs_keys = ["current_player_played_dev", "current_player_hidden_dev",
                                  "next_player_played_dev", "next_next_player_played_dev",
                                  "next_next_next_player_played_dev"]

        self.consider_all_moves_for_opening_placement = consider_all_moves_for_opening_placement
        self.dont_propose_devcards = dont_propose_devcards
        self.dont_propose_trades = dont_propose_trades
        self.zero_opponent_hidden_states = zero_opponent_hidden_states
        self.sample_actions_fn = sample_actions_fn
        self.max_init_actions = max_init_actions
        self.max_depth = max_depth
        self.max_thinking_time = max_thinking_time
        self.num_subprocesses = num_subprocesses
        self.gamma = gamma

        self.value_moving_average = MovingAvgCalculator(window_size=500)

        if subprocess_start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(num_subprocesses)])
        self.processes = []

        self.shared_queue = ctx.Queue()

        for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            args = (work_remote, remote, self.shared_queue, i)
            process = ctx.Process(target=worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

    def initialise_policy(self):
        #intialise policies
        for remote in self.remotes:
            remote.send(("initialise_policy", CloudpickleWrapper((self.base_policy.state_dict(), self.player_id, self.gamma))))
        confirmation = [remote.recv() for remote in self.remotes]

    def act(self, curr_obs, curr_hidden_states, curr_env_state, curr_action_masks, initial_settlement=False, decision_no=0):
        self.proposed_actions, self.player_next_hidden_states = self.sample_actions_fn(
            curr_obs, curr_hidden_states[self.player_id], curr_action_masks,
            self.base_policy, self.max_init_actions,
            consider_all_initial_settlements = self.consider_all_moves_for_opening_placement,
            initial_settlement_phase=initial_settlement, dont_propose_devcards=self.dont_propose_devcards,
            dont_propose_trades=self.dont_propose_trades
        )

        if self.zero_opponent_hidden_states:
            for id in [PlayerId.Blue, PlayerId.Red, PlayerId.Orange, PlayerId.White]:
                if id != self.player_id:
                    curr_hidden_states[id] = (
                        torch.zeros(1, self.base_policy.lstm_size, device="cpu"),
                        torch.zeros(1, self.base_policy.lstm_size, device="cpu")
                    )

        if(len(self.proposed_actions) == 1):
            return self.proposed_actions[0], self.player_next_hidden_states[0]

        thinking_time = self.max_thinking_time * (len(self.proposed_actions) / self.max_init_actions)

        self.workers_ready_to_simulate = deque(range(self.num_subprocesses))

        self.num_simulations_finished = 0
        self.num_simulations_in_progress = 0
        self.num_simulations_finished_each_action = np.zeros((len(self.proposed_actions), ))
        self.num_simulations_started_each_action = np.zeros((len(self.proposed_actions), ))
        self.exploit_scores = np.zeros((len(self.proposed_actions), ))

        for remote in self.remotes:
            remote.send(("set_start_states", CloudpickleWrapper((curr_env_state, curr_hidden_states, curr_obs))))
        confirmation = [remote.recv() for remote in self.remotes]
        for remote in self.remotes:
            remote.send(("set_initial_actions", CloudpickleWrapper((self.proposed_actions, self.player_next_hidden_states))))
        confirmation = [remote.recv() for remote in self.remotes]

        start_time = time.time()
        elapsed_time = 0.0

        while elapsed_time < thinking_time:
        # while self.num_simulations_finished < 100: #deterministic testing
            while_loop_count = 0
            while len(self.workers_ready_to_simulate) > 0:
                worker_id = self.workers_ready_to_simulate.pop()
                action_id = self._select_action()
                self.remotes[worker_id].send(("run_simulation", (action_id, self.max_depth)))
                confirmation = self.remotes[worker_id].recv()
                self.num_simulations_in_progress += 1
                self.num_simulations_started_each_action[action_id] += 1

                while_loop_count += 1
                if while_loop_count > len(self.remotes):
                    break

            while_loop_count = 0
            while not self.shared_queue.empty():
                pred_val, ac_id, worker_id = self.shared_queue.get()
                # print("action: {}. pred_val: {}".format(ac_id, pred_val))
                self._update_stats(pred_val, ac_id)
                self.workers_ready_to_simulate.append(worker_id)

                while_loop_count += 1
                if while_loop_count > len(self.remotes):
                    break

            elapsed_time = time.time() - start_time

        #wait for all envs to finish their final runs
        while len(self.workers_ready_to_simulate) < self.num_subprocesses:
            while not self.shared_queue.empty():
                pred_val, ac_id, worker_id = self.shared_queue.get()
                self._update_stats(pred_val, ac_id)
                self.workers_ready_to_simulate.append(worker_id)

        best_action_id = self._select_action(explore=False)
        # print("\nDecision: {}. Action id: {}. value for best action: {:.2f} (num times selected: {})\n".format(decision_no, best_action_id, self.exploit_scores[best_action_id] / self.num_simulations_finished_each_action[best_action_id], self.num_simulations_finished_each_action[best_action_id]))
        return self.proposed_actions[best_action_id], self.player_next_hidden_states[best_action_id]

    def _select_action(self, explore=True):
        best_score = -np.inf
        best_action = None

        for i in range(len(self.proposed_actions)):
            exploit_score = self.exploit_scores[i] / (self.num_simulations_finished_each_action[i] + 1e-5)
            explore_score = math.sqrt(2.0 * math.log(self.num_simulations_finished + 2) / (self.num_simulations_finished_each_action[i] + self.num_simulations_started_each_action[i] + 1e-10))
            if explore:
                score_std = max(self.value_moving_average.get_std(), 1.0)
                score = exploit_score + 2.0 * score_std * explore_score
            else:
                score = exploit_score

            if score > best_score:
                best_score = score
                best_action = i

        return best_action

    def _update_stats(self, val, action_id):
        self.num_simulations_in_progress -= 1
        self.num_simulations_started_each_action[action_id] -= 1
        self.num_simulations_finished += 1
        self.num_simulations_finished_each_action[action_id] += 1

        self.exploit_scores[action_id] += val
        self.value_moving_average.update(val)

    def eval(self):
        return

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