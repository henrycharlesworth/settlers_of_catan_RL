"""
Code based largely on transformer implementation here:
https://github.com/gordicaleksa/pytorch-original-transformer/blob/main/The%20Annotated%20Transformer%20%2B%2B.ipynb
"""

import torch.nn as nn
import numpy as np

from RL.models.multi_headed_attention import MultiHeadedAttention
from RL.models.utils import get_clones

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class FeedForwardNet(nn.Module):
    def __init__(self, model_dimension, width_mult):
        super().__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.linear1 = init_(nn.Linear(model_dimension, width_mult * model_dimension))
        self.linear2 = init_(nn.Linear(width_mult * model_dimension, model_dimension))

        self.relu = nn.ReLU()

    def forward(self, rep_batch):
        return self.linear2(self.relu(self.linear1(rep_batch)))


class SubLayerLogic(nn.Module):
    def __init__(self, model_dimension):
        super(SubLayerLogic, self).__init__()
        self.norm = nn.LayerNorm(model_dimension)

    def forward(self, rep_batch, sublayer_module):
        #residual connection
        return rep_batch + sublayer_module(self.norm(rep_batch))


class EncoderLayer(nn.Module):
    def __init__(self, model_dimension, multi_headed_attention, pointwise_net):
        super(EncoderLayer, self).__init__()
        self.sublayers = get_clones(SubLayerLogic(model_dimension), 2)

        self.multi_headed_attention = multi_headed_attention
        self.pointwise_net = pointwise_net

        self.model_dimension = model_dimension

    def forward(self, rep_batch):
        encoder_self_attention = lambda rb: self.multi_headed_attention(query=rb, key=rb, value=rb, mask=None)

        rep_batch = self.sublayers[0](rep_batch, encoder_self_attention)
        return self.sublayers[1](rep_batch, self.pointwise_net)


class TileEncoder(nn.Module):
    def __init__(self, tile_in_dim, model_dimension, num_heads, num_layers, out_proj_dim):
        super(TileEncoder, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.first_layer = init_(nn.Linear(tile_in_dim, model_dimension))

        mha = MultiHeadedAttention(model_dimension, num_heads)
        ffn = FeedForwardNet(model_dimension, width_mult=2)
        encoder_layer = EncoderLayer(model_dimension, mha, ffn)

        self.encoder_layers = get_clones(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(out_proj_dim)
        self.norm_2 = nn.LayerNorm(model_dimension)

        self.out_proj = init_(nn.Linear(model_dimension, out_proj_dim))
        self.relu = nn.ReLU()


    def forward(self, tile_representations):
        """
        tile_representations should be (batch, num_tiles, rep_dim)
        """
        batch_size = tile_representations.shape[0]

        tile_representations = self.relu(self.norm_2(self.first_layer(tile_representations)))
        for encoder_layer in self.encoder_layers:
            tile_representations = encoder_layer(tile_representations)

        tile_rep_final = self.norm(self.out_proj(tile_representations))
        return self.relu(tile_rep_final.reshape(batch_size, -1))