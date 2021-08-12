import copy
import numpy as np


def run_evaluation_protocol(evaluation_manager, central_policy, earlier_policies, random_policy, args, update_num):

    policies_to_play_against = [random_policy]
    opponent_policy_ids = ["random"]
    # if len(earlier_policies) >= 25:
    #     policies_to_play_against.append(earlier_policies[-25])
    #     opponent_policy_ids.append("25 updates ago")
    # if len(earlier_policies) >= 50:
    #     policies_to_play_against.append(earlier_policies[-50])
    #     opponent_policy_ids.append("500 updates ago")
    # if len(earlier_policies) >= 75:
    #     policies_to_play_against.append(earlier_policies[-75])
    #     opponent_policy_ids.append("75 updates ago")
    # if len(earlier_policies) >= 100:
    #     policies_to_play_against.append(earlier_policies[-100])
    #     opponent_policy_ids.append("1000 updates ago")

    log = {"update": update_num}

    print_str = "\n\n---------------------- EVALUATION (after {} updates) ----------------------\n".format(
        update_num
    )

    start_device = central_policy.dummy_param.device
    central_policy.to("cpu")

    for i, policy in enumerate(policies_to_play_against):
        policies = [copy.deepcopy(central_policy.state_dict()), copy.deepcopy(policies_to_play_against[i]),
                    copy.deepcopy(policies_to_play_against[i]), copy.deepcopy(policies_to_play_against[i])]

        evaluation_manager.update_policies(policies)

        results = evaluation_manager.run_evaluation_episodes(args.num_eval_episodes)
        results = list(zip(*results))

        winners = np.concatenate(results[0])
        game_lengths = np.concatenate(results[1])
        victory_points = np.concatenate(results[2])
        policy_steps = np.concatenate(results[3])

        log[opponent_policy_ids[i]] = {
            "policy_win_frac": np.mean(winners == 0),
            "avg_game_length": np.mean(game_lengths),
            "avg_policy_decisions": np.mean(policy_steps),
            "avg_victory_points": np.mean(victory_points)
        }

        opponent_str = "random" if i == 0 else "policy from "+opponent_policy_ids[i]
        print_str += "{} games against {}. Policy won {}/{}. Avg. game length: {}. Avg num policy decisions: {}. Avg victory points for policy: {}. \n".format(
            args.num_eval_episodes, opponent_str, int(np.sum(winners == 0)), args.num_eval_episodes,
            np.mean(game_lengths), np.mean(policy_steps), np.mean(victory_points)
        )

    print_str += "\n"

    central_policy.to(start_device)

    return log, print_str