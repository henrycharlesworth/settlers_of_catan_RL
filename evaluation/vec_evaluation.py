import torch
import random
from collections import defaultdict

import multiprocessing as mp

from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper

def _worker(
        remote: mp.connection.Connection, parent_remote: mp.connection.Connection, manager_fn_wrapper: CloudpickleWrapper
) -> None:
    parent_remote.close()

    torch.set_num_threads(1)

    evaluation_manager = manager_fn_wrapper.var()

    policy_state_dicts = {}


    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "initialise_policy_pool":
                policy_ids = data
                base_file_name = "../RL/results/just_policy_default_after_update_"
                for id in policy_ids:
                    file_name = base_file_name + str(id) + ".pt"
                    state_dict = torch.load(file_name, map_location="cpu")
                    policy_state_dicts[id] = state_dict
                remote.send(True)
            elif cmd == "run_eval_episodes":
                num_episodes, player_id, other_player_ids = data
                randomise_opponent_policies_each_episode = False
                if other_player_ids is None:
                    randomise_opponent_policies_each_episode = True
                winners = []
                num_game_steps = []
                victory_points_all = []
                policy_steps = []
                entropies = []
                action_types_all = defaultdict(lambda: 0)

                if randomise_opponent_policies_each_episode == False:
                    policies = [policy_state_dicts[player_id],
                                policy_state_dicts[other_player_ids[0]],
                                policy_state_dicts[other_player_ids[1]],
                                policy_state_dicts[other_player_ids[2]]]
                    evaluation_manager._update_policies(policies)

                for ep in range(num_episodes):

                    if randomise_opponent_policies_each_episode:
                        policies = [policy_state_dicts[player_id]] + [random.choice(list(policy_state_dicts.values())),
                                                                      random.choice(list(policy_state_dicts.values())),
                                                                      random.choice(list(policy_state_dicts.values()))]
                        evaluation_manager._update_policies(policies)
                    winner, victory_points, total_steps, policy_decisions, entropy, action_types = evaluation_manager.run_evaluation_game()
                    winners.append(winner)
                    num_game_steps.append(total_steps)
                    victory_points_all.append(victory_points)
                    policy_steps.append(policy_decisions)
                    entropies.append(entropy)

                    for key, val in action_types.items():
                        action_types_all[key] += val

                remote.send((winners, num_game_steps, victory_points_all, policy_steps, entropies, dict(action_types_all)))
        except EOFError:
            break


class SubProcEvaluationManager(object):
    def __init__(self, evaluation_manager_fns, start_method=None):
        self.waiting = False
        self.closed = False
        n_processes = len(evaluation_manager_fns)

        if start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_processes)])
        self.processes = []

        for work_remote, remote, evaluation_manager_fn in zip(self.work_remotes, self.remotes, evaluation_manager_fns):
            args = (work_remote, remote, CloudpickleWrapper(evaluation_manager_fn))
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

    def initialise_policy_pool(self, policy_ids):
        for remote in self.remotes:
            remote.send(("initialise_policy_pool", policy_ids))
        results = [remote.recv() for remote in self.remotes]
        return results

    def run_evaluation_episodes_async(self, num_episodes, player_id, opponent_ids):
        for remote in self.remotes:
            remote.send(("run_eval_episodes", (num_episodes, player_id, opponent_ids)))
        self.waiting = True

    def run_evaluation_episodes_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return results

    def run_evaluation_episodes(self, num_episodes, player_id, opponent_ids):
        self.run_evaluation_episodes_async(num_episodes, player_id, opponent_ids)
        return self.run_evaluation_episodes_wait()

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True