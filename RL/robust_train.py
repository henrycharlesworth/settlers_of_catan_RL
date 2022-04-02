import os
import time
import numpy as np
import random
import torch
import copy
import psutil

from collections import deque

import RL.ppo.utils as utils

from RL.ppo.arguments import get_args
from RL.ppo.game_manager import make_game_manager
from RL.ppo.vec_gather_experience import SubProcGameManager
from RL.models.build_agent_model import build_agent_model
from RL.ppo.process_batch import BatchProcessor
from RL.ppo.update_opponent_policies import update_opponent_policies
from RL.ppo.run_evaluation_protocol import run_evaluation_protocol
from RL.ppo.ppo import PPO
from RL.ppo.vec_evaluation import SubProcEvaluationManager
from RL.ppo.evaluation_manager import make_evaluation_manager

update_num, rollout_manager, evaluation_manager = None, None, None
DEBUG = False

def main():
    global update_num, rollout_manager, evaluation_manager

    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    experiment_dir = "RL/results"
    os.makedirs(experiment_dir, exist_ok=True)

    # torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    rollout_manager_fns = [
        make_game_manager(args.num_envs_per_process, args.num_steps) for _ in range(args.num_processes)
    ]
    rollout_manager = SubProcGameManager(rollout_manager_fns)

    central_policy = build_agent_model(device=device)

    if args.load_from_checkpoint:
        central_policy_sd, earlier_policies, eval_logs, start_update, args, entropy_coef, rew_anneal_fac = torch.load("RL/results/"+args.load_file_path)
        update_opponent_policies(earlier_policies, rollout_manager, args)
        central_policy.load_state_dict(central_policy_sd)
        central_policy.to("cpu")
        rollout_manager.update_policy(central_policy.state_dict(), policy_id=0)
        central_policy.to(device)
    else:
        earlier_policies = deque(maxlen=args.num_policies_to_store)
        central_policy.to("cpu")
        earlier_policies.append(copy.deepcopy(central_policy.state_dict()))
        central_policy.to(device)
        start_update = 0
        eval_logs = []
        entropy_coef = args.entropy_coef_start
        rew_anneal_fac = 1.0

    update_num = start_update
    curr_entropy_coef = entropy_coef
    curr_reward_weight = rew_anneal_fac

    random_policy_model = build_agent_model()
    random_policy = copy.deepcopy(random_policy_model.state_dict())
    del random_policy_model

    rollout_storage = BatchProcessor(args, central_policy.lstm_size, device=device)

    agent = PPO(central_policy, args)

    agent.entropy_coef = curr_entropy_coef
    rollout_manager.update_annealing_factor(curr_reward_weight)

    eval_manager_fns = [
        make_evaluation_manager() for _ in range(args.num_eval_processes)
    ]
    evaluation_manager = SubProcEvaluationManager(eval_manager_fns)

    start_time = time.time()
    num_updates = int(args.total_env_steps) // args.num_steps // (args.num_processes * args.num_envs_per_process)
    steps_per_update = int(args.num_steps * args.num_processes * args.num_envs_per_process)

    def run_update():
        global update_num, curr_entropy_coef, curr_reward_weight

        if args.use_linear_lr_decay:
            utils.update_linear_schedule(agent.optimiser, update_num, num_updates, args.lr)

        rollouts = rollout_manager.gather_rollouts()
        rollout_storage.process_rollouts(rollouts)

        val_loss, action_loss, entropy_loss = agent.update(rollout_storage)

        central_policy.to("cpu")
        rollout_manager.update_policy(central_policy.state_dict(), policy_id=0)
        central_policy.to(device)

        #anneal dense reward/ entropy coef
        if update_num > args.entropy_coef_start_anneal and update_num <= args.entropy_coef_end_anneal:
            start_update = args.entropy_coef_start_anneal
            end_update = args.entropy_coef_end_anneal
            start_value = args.entropy_coef_start
            end_value = args.entropy_coef_final

            value = start_value + ((update_num - start_update) / (end_update - start_update)) * (end_value - start_value)
            agent.entropy_coef = value
            curr_entropy_coef = value

        if update_num > args.dense_reward_anneal_start and update_num <= args.dense_reward_anneal_end:
            start_update = args.dense_reward_anneal_start
            end_update = args.dense_reward_anneal_end
            value = 1.0 + ((update_num - start_update) / (end_update - start_update)) * (0.0 - 1.0)
            rollout_manager.update_annealing_factor(value)
            curr_reward_weight = value

        t_current = time.time()
        print(
            "Updates complete: {}. Total policy steps: {}. Number of games complete: {}. Total elapsed time: {} hours.".format(
                update_num+1, steps_per_update * (update_num+1), rollout_storage.games_complete,
                            (t_current - start_time) / 3600.0
            ))

        if update_num % args.add_policy_every == 0 and update_num > 0:
            central_policy.to("cpu")
            earlier_policies.append(copy.deepcopy(central_policy.state_dict()))
            central_policy.to(device)

        if update_num % args.update_opponent_policies_every == 0:
            update_opponent_policies(earlier_policies, rollout_manager, args)

        if update_num % args.eval_every == 0 and update_num > 0:
            log, print_summary = run_evaluation_protocol(evaluation_manager, central_policy, earlier_policies,
                                                         random_policy, args, update_num, curr_entropy_coef, curr_reward_weight)
            eval_logs.append(log)

            print(print_summary)

            torch.save(central_policy.state_dict(),
                       "RL/results/" + args.expt_id + "_after_update_" + str(update_num) + ".pt")

        update_num += 1

        torch.save((central_policy.state_dict(), earlier_policies, eval_logs, update_num, args),
                   "RL/results/current.pt")

    def fail_handler():
        print("Error or update exceeded specified timeout (something broke). Attempting to reinitialise everything and continue training!")

        global rollout_manager, evaluation_manager
        for process in rollout_manager.processes:
            process.kill()
        for process in evaluation_manager.processes:
            process.kill()

        rollout_manager = SubProcGameManager(rollout_manager_fns)
        evaluation_manager = SubProcEvaluationManager(eval_manager_fns)

        central_policy.to("cpu")
        rollout_manager.update_policy(central_policy.state_dict(), policy_id=0)
        central_policy.to(device)
        update_opponent_policies(earlier_policies, rollout_manager, args)

        print("Environments reinitialised - continuing training.")


    while update_num < num_updates:

        run_update()

        if psutil.virtual_memory().percent > 95.0:
            #stupid memory leak - need to restart everything using a bash script as a workaround... actually should be fixed now but meh
            fail_handler()
            os.system('kill %d' % os.getpid())


if __name__ == "__main__":
    main()
