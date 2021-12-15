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
    game_manager = manager_fn_wrapper.var()

    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "gather_rollouts":
                observations, hidden_states, rewards, actions, action_masks, \
                    action_log_probs, dones = game_manager.gather_rollouts()
                game_manager._after_rollouts()
                remote.send(
                    CloudpickleWrapper((observations, hidden_states, rewards, actions, action_masks,
                                        action_log_probs, dones))
                )
            elif cmd == "reset":
                game_manager.reset()
                remote.send(True)
            elif cmd == "close":
                remote.close()
                break
            elif cmd == "update_policy":
                state_dict = data[0].var
                policy_id = data[1]
                game_manager._update_policy(state_dict, policy_id)
                remote.send(True)
            elif cmd == "update_annealing_factor":
                game_manager._update_annealing_factor(data)
                remote.send(True)
            elif cmd == "seed":
                np.random.seed(data)
                random.seed(data)
                torch.manual_seed(data)
                remote.send(True)
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class SubProcGameManager(object):
    """
    Analagous to SubProcVecEnv from stable baselines, but managing multiple games per worker and the policies of each
    player live within the worker too. Gathers the full rollouts for the active players rather than a single step.
    """

    def __init__(self, game_manager_fns, start_method = None):
        self.waiting = False
        self.closed = False
        n_processes = len(game_manager_fns)

        if start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_processes)])
        self.processes = []

        for work_remote, remote, game_manager_fn in zip(self.work_remotes, self.remotes, game_manager_fns):
            args = (work_remote, remote, CloudpickleWrapper(game_manager_fn))
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

    def gather_async(self):
        for remote in self.remotes:
            remote.send(("gather_rollouts", None))
        self.waiting = True

    def gather_wait(self):
        results = [remote.recv().var for remote in self.remotes]
        self.waiting = False
        """manage processing rollouts outside of here"""
        return results

    def gather_rollouts(self):
        self.gather_async()
        return self.gather_wait()

    def update_policy(self, state_dict, process_id = None, policy_id = 0):
        if process_id is None:
            for remote in self.remotes:
                remote.send(("update_policy", (CloudpickleWrapper(state_dict), policy_id)))
            results = [remote.recv() for remote in self.remotes]
        else:
            self.remotes[process_id].send(("update_policy", (CloudpickleWrapper(state_dict), policy_id)))
            results = self.remotes[process_id].recv()
        return results

    def update_annealing_factor(self, annealing_factor):
        for remote in self.remotes:
            remote.send(("update_annealing_factor", annealing_factor))
        results = [remote.recv() for remote in self.remotes]
        return results

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        return results

    def seed(self, seeds):
        for i, remote in enumerate(self.remotes):
            remote.send(("seed", seeds[i]))
        results = [remote.recv() for remote in self.remotes]
        return results