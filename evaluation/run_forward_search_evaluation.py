import numpy as np
import torch
import argparse
from pathlib import Path
from collections import defaultdict
import random

from evaluation.evaluation_manager import EvaluationManager
from RL.models.build_agent_model import build_agent_model
from RL.forward_search_policy.policy import ForwardSearchPolicy
from RL.forward_search_policy.sample_actions_fn import default_sample_actions

parser = argparse.ArgumentParser()
parser.add_argument('--base-policy-file', type=str, required=True)
parser.add_argument('--max-thinking-time', type=float, default=10.0)
parser.add_argument('--gamma', type=float, default=0.999)
parser.add_argument('--num-subprocesses', type=int, default=32)
parser.add_argument('--num-games', type=int, default=100)
parser.add_argument('--max-init-actions', type=int, default=10)
parser.add_argument('--max-depth', type=int, default=15)
parser.add_argument('--consider-all-moves-for-opening-placements', action='store_true',default=False)
parser.add_argument('--dont-propose-devcards', action='store_true', default=False)
parser.add_argument('--dont-propose-trades', action='store_true', default=False)
parser.add_argument('--zero-opponent-hidden-states', action='store_true', default=False)
parser.add_argument('--other-policies', type=str, default="")

args = parser.parse_args()

torch.manual_seed(10)
np.random.seed(10)
random.seed(10)

if __name__ == "__main__":
    policy_state_dict = torch.load(Path("RL", "results", args.base_policy_file), map_location="cpu")

    if args.other_policies != "":
        other_policies = args.other_policies.split(" ")
        if len(other_policies) == 1:
            other_policies = [other_policies[0]] * 3
        other_policy_state_dicts = [
            torch.load(Path("RL", "results", other_policies[i]), map_location="cpu") for i in range(3)
        ]
    else:
        other_policy_state_dicts = [policy_state_dict, policy_state_dict, policy_state_dict]

    forward_search_policy = ForwardSearchPolicy(policy_state_dict, default_sample_actions, args.max_init_actions,
                                                args.max_depth, args.max_thinking_time, gamma=args.gamma,
                                                num_subprocesses=args.num_subprocesses,
                                                zero_opponent_hidden_states=args.zero_opponent_hidden_states,
                                                consider_all_moves_for_opening_placement=args.consider_all_moves_for_opening_placements,
                                                dont_propose_trades=args.dont_propose_trades,
                                                dont_propose_devcards=args.dont_propose_devcards)
    policies = [forward_search_policy, build_agent_model(), build_agent_model(), build_agent_model()]
    for i in range(1, 4):
        policies[i].load_state_dict(other_policy_state_dicts[i-1])

    evaluation_manager = EvaluationManager(policies=policies)

    winners_all = []
    num_game_steps_all = []
    victory_points_all = []
    forward_policy_decisions_all = []
    action_types_all = defaultdict(lambda: 0)

    for i in range(args.num_games):
        winner, victory_points, total_steps, policy_decisions, _, action_types, _, _ = evaluation_manager.run_evaluation_game()

        winners_all.append(winner)
        num_game_steps_all.append(total_steps)
        victory_points_all.append(victory_points)
        forward_policy_decisions_all.append(policy_decisions)

        for key, val in action_types.items():
            action_types_all[key] += val

        print("Game {} finished. Winner: {}. Fraction of games won by forward search so far: {}".format(
            i+1, winner, np.mean(np.array(winners_all)==0)
        ))

        torch.save((
            winners_all, num_game_steps_all, victory_points_all, forward_policy_decisions_all, sorted(dict(action_types_all).items())
        ), "forward_policy_evaluation.pt")