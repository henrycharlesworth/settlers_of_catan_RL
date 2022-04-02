import joblib
import numpy as np
import time
import os
import argparse
from collections import defaultdict

from evaluation.evaluation_manager import make_evaluation_manager
from evaluation.vec_evaluation import SubProcEvaluationManager

parser = argparse.ArgumentParser()
parser.add_argument('--evaluate-every-nth-policy', type=int, default=4)
parser.add_argument('--evaluation-games-per-policy', type=int, default=320)
parser.add_argument('--evaluation-type', type=str, default="previous_policies",
                    choices=["previous_policies", "random"])
parser.add_argument('--previous-shift', type=int, default=5)

args = parser.parse_args()

NUM_PROCESSES = 32
DETAILED_LOGS = False

if __name__ == "__main__":
    policy_files = os.listdir("../RL/results/")
    policy_files = [file for file in policy_files if file.startswith("default")]
    policy_file_ids = [int(file[:-3].split("_")[-1]) for file in policy_files]
    policy_file_ids.sort()

    if args.evaluation_type == "previous_policies":
        first_id_idx = args.previous_shift
    else:
        first_id_idx = 0

    policy_id_idxs = np.arange(first_id_idx, len(policy_file_ids), args.evaluate_every_nth_policy)
    policies_to_evaluate_ids = policy_file_ids[first_id_idx::args.evaluate_every_nth_policy]

    results = {}

    eval_manager_fns = [
        make_evaluation_manager() for _ in range(NUM_PROCESSES)
    ]
    evaluation_manager = SubProcEvaluationManager(eval_manager_fns)
    evaluation_manager.initialise_policy_pool(policy_file_ids, DETAILED_LOGS)

    for i, player_id in enumerate(policies_to_evaluate_ids):
        t1 = time.time()

        if args.evaluation_type == "previous_policies":
            prev_policy_id = policy_file_ids[policy_id_idxs[i] - args.previous_shift]
            opponent_policy_ids = [prev_policy_id, prev_policy_id, prev_policy_id]
        elif args.evaluation_type == "random":
            opponent_policy_ids = None
        else:
            raise NotImplementedError

        res = evaluation_manager.run_evaluation_episodes(
            args.evaluation_games_per_policy // NUM_PROCESSES,
            player_id, opponent_policy_ids
        )
        res = list(zip(*res))

        action_types = defaultdict(lambda: 0)

        winners = np.concatenate(res[0])
        game_lengths = np.concatenate(res[1])
        victory_points = np.concatenate(res[2])
        policy_steps = np.concatenate(res[3])
        entropies = np.concatenate(res[4])

        for action_type in res[5]:
            for key, val in action_type.items():
                action_types[key] += val

        for i in range(12):
            action_types[i] = action_types.get(i, 0)

        type_log_prob_tuples = []
        for l1 in res[6]:
            for l2 in l1:
                type_log_prob_tuples += l2

        type_prob_dict = defaultdict(lambda: 0)
        type_prob_count = defaultdict(lambda: 0)
        type_prob_sum = defaultdict(lambda: 0)
        for entry in type_log_prob_tuples:
            type_prob_count[entry[0]] += 1
            type_prob_sum[entry[0]] += np.exp(entry[1])
            type_prob_dict[entry[0]] = type_prob_sum[entry[0]] / type_prob_count[entry[0]]

        # values = np.concatenate(np.concatenate(res[7]))

        results[player_id] = {
            "win_frac": np.mean(winners == 0),
            "avg_game_length": np.mean(game_lengths),
            "avg_pol_decisions": np.mean(policy_steps),
            "avg_vps": np.mean(victory_points),
            "draw_frac": np.mean(winners == -1),
            "avg_entropy": np.mean(entropies),
            "action_types": sorted(dict(action_types).items()),
            "type_log_probs": type_log_prob_tuples,
        }

        if DETAILED_LOGS:
            detailed_head_logs = []
            for l1 in res[8]:
                for l2 in l1:
                    detailed_head_logs += l2
            results[player_id]["detailed_head_logs"] = detailed_head_logs

        print("{} games for policy after {} updates completed in {} seconds!".format(
            args.evaluation_games_per_policy,
            player_id,
            time.time() - t1
        ))

        print('-----------------    WIN FRAC: {}'.format(results[player_id]["win_frac"]))
        print('-----------------    AVG GAME LENGTH: {}'.format(results[player_id]["avg_game_length"]))
        print('-----------------    AVG POLICY DECISIONS: {}'.format(results[player_id]["avg_pol_decisions"]))
        print('-----------------    AVG VICTORY POINTS: {}'.format(results[player_id]["avg_vps"]))
        print('-----------------    AVG ENTROPY: {}'.format(results[player_id]["avg_entropy"]))
        print('-----------------    DRAW FRACTION: {}'.format(results[player_id]["draw_frac"]))
        print("action types:")
        print(sorted(action_types.items()))
        print("\n")
        print("avg action probs sorted by type:")
        print(sorted(type_prob_dict.items()))
        print("\n")

        joblib.dump(results, "evaluation_results.pt")
