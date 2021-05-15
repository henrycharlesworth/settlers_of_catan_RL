import numpy as np
import random
import torch

import multiprocessing as mp

from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper

def _worker(
        remote: mp.connection.Connection, parent_remote: mp.connection.Connection, manager_fn_wrapper: CloudpickleWrapper
) -> None:
    parent_remote.close()

    torch.set_num_threads(1)
    evaluation_manager = manager_fn_wrapper.var()

    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "run_eval_episodes":
                num_episodes = data
                winners = []
                num_game_steps = []
                victory_points_all = []
                policy_steps = []
                for ep in range(num_episodes):
                    winner, victory_points, total_steps, policy_decisions = evaluation_manager.run_evaluation_game()
                    winners.append(winner)
                    num_game_steps.append(total_steps)
                    victory_points_all.append(victory_points)
                    policy_steps.append(policy_decisions)
                remote.send((winners, num_game_steps, victory_points_all, policy_steps))
            elif cmd == "update_policies":
                state_dicts = data.var
                evaluation_manager._update_policies(state_dicts)
                remote.send(True)
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

    def run_evaluation_episodes_async(self, total_episodes):
        eps_per_process = total_episodes // len(self.processes)
        for remote in self.remotes:
            remote.send(("run_eval_episodes", eps_per_process))
        self.waiting = True

    def run_evaluation_episodes_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return results

    def run_evaluation_episodes(self, total_episodes):
        self.run_evaluation_episodes_async(total_episodes)
        return self.run_evaluation_episodes_wait()

    def update_policies(self, state_dicts):
        for remote in self.remotes:
            remote.send(("update_policies", CloudpickleWrapper(state_dicts)))
        results = [remote.recv() for remote in self.remotes]
        return results

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