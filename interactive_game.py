import torch

from game.enums import PlayerId
from env.wrapper import EnvWrapper
from ui.display import Display
from RL.models.build_agent_model import build_agent_model

device = "cpu"

policy_1_sd, _, _, _, _ = torch.load("RL/results/default_after_update_25.pt", map_location=device)
policy_2_sd, _, _, _, _ = torch.load("RL/results/default_after_update_25.pt", map_location=device)
policy_3_sd, _, _, _, _ = torch.load("RL/results/default_after_update_25.pt", map_location=device)
policy_1 = build_agent_model(device=device)
policy_2 = build_agent_model(device=device)
policy_3 = build_agent_model(device=device)
policy_1.load_state_dict(policy_1_sd)
policy_2.load_state_dict(policy_2_sd)
policy_3.load_state_dict(policy_3_sd)

policies = {
    PlayerId.White: "human",
    PlayerId.Red: policy_1,
    PlayerId.Orange: policy_2,
    PlayerId.Blue: policy_3
}

env = EnvWrapper(policies=policies)
env.reset()
display = Display(env=env, game=env.game, interactive=True, policies=policies, test=True)