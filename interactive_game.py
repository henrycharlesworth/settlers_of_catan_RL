import torch

from game.enums import PlayerId
from env.wrapper import EnvWrapper
from ui.display import Display
from RL.models.build_agent_model import build_agent_model
from RL.forward_search_policy.policy import ForwardSearchPolicy
from RL.forward_search_policy.sample_actions_fn import default_sample_actions

device = "cpu"

if __name__ == "__main__":
    # policy_1_sd = torch.load("RL/results/default_after_update_1050.pt", map_location=device)
    # policy_2_sd = torch.load("RL/results/default_after_update_1050.pt", map_location=device)
    # policy_3_sd = torch.load("RL/results/default_after_update_1050.pt", map_location=device)
    # policy_4_sd = torch.load("RL/results/default_after_update_1050.pt", map_location=device)
    # policy_1 = build_agent_model(device=device)
    # policy_2 = build_agent_model(device=device)
    # policy_3 = build_agent_model(device=device)
    # policy_1.load_state_dict(policy_1_sd)
    # policy_2.load_state_dict(policy_2_sd)
    # policy_3.load_state_dict(policy_3_sd)
    #
    # policy_4 = ForwardSearchPolicy(policy_4_sd, default_sample_actions, 10, max_depth=15,
    #                                max_thinking_time=10, gamma=0.999, num_subprocesses=11,
    #                                zero_opponent_hidden_states=True, consider_all_moves_for_opening_placement=True,
    #                                player_id=PlayerId.Blue)
    # policy_4.initialise_policy()

    # policies = {
    #     PlayerId.White: "human",
    #     PlayerId.Red: policy_1,
    #     PlayerId.Orange: policy_2,
    #     PlayerId.Blue: policy_4
    # }
    # policies = {
    #     PlayerId.White: policy_4,
    #     PlayerId.Red: policy_1,
    #     PlayerId.Orange: policy_2,
    #     PlayerId.Blue: policy_3
    # }

    policies = {
        PlayerId.White: "human",
        PlayerId.Red: "human",
        PlayerId.Orange: "human",
        PlayerId.Blue: "human"
    }

    env = EnvWrapper(policies=policies)
    env.reset()
    display = Display(env=env, game=env.game, interactive=True, policies=policies, test=False, debug_mode=False)