"""
Process observation from the environment (ready to be fed into the LSTM)
"""

import torch
import torch.nn as nn
import numpy as np

from RL.models.multi_headed_attention import MultiHeadedAttention
from RL.models.tile_encoder import TileEncoder
from RL.models.player_modules import CurrentPlayerModule, OtherPlayersModule

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class ObservationModule(nn.Module):
    def __init__(self, tile_in_dim, tile_model_dim, curr_player_main_in_dim, other_player_main_in_dim,
                 dev_card_embed_dim, dev_card_model_dim, observation_out_dim, tile_model_num_heads=4, proj_dev_card_dim=25,
                 dev_card_model_num_heads=4, tile_encoder_num_layers=2, proj_tile_dim=25):
        super(ObservationModule, self).__init__()
        self.tile_in_dim = tile_in_dim
        self.tile_model_dim = tile_model_dim
        self.curr_player_main_in_dim = curr_player_main_in_dim
        self.dev_card_embed_dim = dev_card_embed_dim
        self.dev_card_model_dim = dev_card_model_dim
        self.tile_model_num_heads = tile_model_num_heads
        self.proj_dev_card_dim = proj_dev_card_dim
        self.dev_card_model_num_heads = dev_card_model_num_heads

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.dev_card_embedding = nn.Embedding(6, dev_card_embed_dim)
        self.hidden_card_mha = MultiHeadedAttention(dev_card_model_dim, dev_card_model_num_heads)
        self.played_card_mha = MultiHeadedAttention(dev_card_model_dim, dev_card_model_num_heads)

        self.tile_encoder = TileEncoder(tile_in_dim, tile_model_dim, tile_model_num_heads, tile_encoder_num_layers,
                                        proj_tile_dim)

        self.current_player_module = CurrentPlayerModule(curr_player_main_in_dim, dev_card_embed_dim,
                                                         dev_card_model_dim, proj_dev_card_dim)
        self.other_players_module = OtherPlayersModule(other_player_main_in_dim, dev_card_embed_dim,
                                                       dev_card_model_dim, proj_dev_card_dim)

        self.final_layer = init_(nn.Linear(19 * proj_tile_dim + 4 * 128, observation_out_dim))
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(observation_out_dim)

    def forward(self, obs_dict):
        tile_encodings = self.tile_encoder(obs_dict["tile_representations"])

        current_player_output = self.current_player_module(obs_dict["current_player_main"],
                                    obs_dict["current_player_hidden_dev"], obs_dict["current_player_played_dev"],
                                    self.dev_card_embedding, self.hidden_card_mha, self.played_card_mha)
        other_player_outputs = [self.other_players_module(obs_dict[o_player+"_player_main"],
                obs_dict[o_player+"_player_played_dev"], self.dev_card_embedding, self.played_card_mha) \
                                for o_player in ["next", "next_next", "next_next_next"]]

        final_input = torch.cat((tile_encodings, current_player_output, *other_player_outputs), dim=-1)
        return self.relu(self.norm(self.final_layer(final_input)))