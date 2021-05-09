"""
Code based largely on implementation here:
https://github.com/gordicaleksa/pytorch-original-transformer/blob/main/The%20Annotated%20Transformer%20%2B%2B.ipynb
"""

import math
import torch
import torch.nn as nn

from RL.models.utils import get_clones

class MultiHeadedAttention(nn.Module):
    def __init__(self, model_dimension, num_heads):
        super(MultiHeadedAttention, self).__init__()
        assert model_dimension % num_heads == 0
        """
        inputs should be (batch, max_seq_len, model_dim)
        """
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.head_dimension = int(model_dimension / num_heads)
        self.num_heads = num_heads

        self.qkv_nets = get_clones(nn.Linear(model_dimension, model_dimension), 3)
        self.out_proj_net = nn.Linear(model_dimension, model_dimension)

        self.softmax = nn.Softmax(dim=-1)

    def attention(self, query, key, value, mask):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dimension)

        if mask is not None:
            scores.masked_fill_(mask == torch.tensor(False, device=self.dummy_param.device), float("-inf"))

        attention_weights = self.softmax(scores)

        intermediate_reps = torch.matmul(attention_weights, value)

        return intermediate_reps

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]

        # step 1: input linear projection
        # shape goes from (B, T, NH*ND) to (B, NH, T, HD)
        query, key, value = [net(x).view(batch_size, -1, self.num_heads, self.head_dimension).transpose(1, 2)
                             for net, x in zip(self.qkv_nets, (query, key, value))]

        #step 2: apply attention
        intermediate_reps = self.attention(query, key, value, mask)

        #step 3: reshape into (B, T, NH*ND)
        intermediate_reps = intermediate_reps.transpose(1, 2).reshape(batch_size, -1,
                                                                      self.num_heads * self.head_dimension)

        return self.out_proj_net(intermediate_reps)