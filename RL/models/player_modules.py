import torch
import torch.nn as nn
import numpy as np

from torch.nn.utils.rnn import pad_sequence

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class CurrentPlayerModule(nn.Module):
    def __init__(self, main_input_dim, dev_card_embed_dim, dev_card_model_dim, proj_dev_card_dim):
        super(CurrentPlayerModule, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.main_input_dim = main_input_dim
        self.dev_card_embed_dim = dev_card_embed_dim
        self.dev_card_model_dim = dev_card_model_dim
        self.proj_dev_card_dim = proj_dev_card_dim

        self.main_input_layer_1 = init_(nn.Linear(main_input_dim, 256))
        self.relu = nn.ReLU()

        self.norm = nn.LayerNorm(dev_card_model_dim)
        self.norm_1 = nn.LayerNorm(256)
        self.norm_2 = nn.LayerNorm(proj_dev_card_dim)
        self.norm_3 = nn.LayerNorm(proj_dev_card_dim)
        self.norm_4 = nn.LayerNorm(128)

        self.proj_hidden_dev_card = init_(nn.Linear(dev_card_model_dim, proj_dev_card_dim))
        self.proj_played_dev_card = init_(nn.Linear(dev_card_model_dim, proj_dev_card_dim))

        self.final_linear_layer = init_(nn.Linear(2 * proj_dev_card_dim + 256, 128))


    def forward(self, main_input, hidden_dev_cards, played_dev_cards, dev_card_embedding, hidden_card_mha,
                played_card_mha):
        """
        for now assuming hidden_dev_cards is variable length list of tensors of integers.
        index 0 will correspond to none/dummy embedding
        """
        if isinstance(hidden_dev_cards, list):
            hidden_dev_cards_lengths = [len(hidden_cards) for hidden_cards in hidden_dev_cards]
            padded_hidden_dev_cards = pad_sequence(hidden_dev_cards, batch_first=True).long()
        else:
            hidden_dev_cards_lengths = (hidden_dev_cards.shape[-1] - (hidden_dev_cards == 0).sum(dim=-1)).cpu().data.numpy()
            hidden_dev_cards_lengths[hidden_dev_cards_lengths == 0] = 1
            hidden_dev_cards_lengths = list(hidden_dev_cards_lengths)
            padded_hidden_dev_cards = hidden_dev_cards

        hidden_dev_card_embeddings = dev_card_embedding(padded_hidden_dev_cards)
        hidden_dev_card_masks = torch.zeros(main_input.shape[0], 1, 1, padded_hidden_dev_cards.shape[1],
                                            device=self.dummy_param.device)
        for b in range(main_input.shape[0]):
            hidden_dev_card_masks[b, ..., :hidden_dev_cards_lengths[b]] = 1.0

        hidden_dev_representations = self.norm(hidden_card_mha(hidden_dev_card_embeddings, hidden_dev_card_embeddings,
                                                                hidden_dev_card_embeddings, hidden_dev_card_masks))
        hidden_dev_card_masks = hidden_dev_card_masks.squeeze(1).transpose(-1, -2).long()
        hidden_dev_representations[hidden_dev_card_masks.repeat(1, 1, hidden_dev_representations.shape[-1]) == 0] = 0.0
        # hidden_dev_out = torch.cat((
        #     hidden_dev_representations.min(dim=1)[0], hidden_dev_representations.max(dim=1)[0],
        #     hidden_dev_representations.sum(dim=1)
        # ), dim=-1)
        hidden_dev_out = hidden_dev_representations.sum(dim=1)
        hidden_dev_out = self.relu(self.norm_2(self.proj_hidden_dev_card(hidden_dev_out)))

        if isinstance(played_dev_cards, list):
            played_dev_card_lengths = [len(played_cards) for played_cards in played_dev_cards]
            padded_played_dev_cards = pad_sequence(played_dev_cards, batch_first=True).long()
        else:
            played_dev_card_lengths = (played_dev_cards.shape[-1] - (played_dev_cards==0).sum(dim=-1)).cpu().data.numpy()
            played_dev_card_lengths[played_dev_card_lengths == 0] = 1
            played_dev_card_lengths = list(played_dev_card_lengths)
            padded_played_dev_cards = played_dev_cards
        played_dev_card_embeddings = dev_card_embedding(padded_played_dev_cards)
        played_dev_card_masks = torch.zeros(main_input.shape[0], 1, 1, padded_played_dev_cards.shape[1],
                                            device=self.dummy_param.device)
        for b in range(main_input.shape[0]):
            played_dev_card_masks[b, ..., :played_dev_card_lengths[b]] = 1.0

        played_dev_representations = self.norm(played_card_mha(played_dev_card_embeddings, played_dev_card_embeddings,
                                                                played_dev_card_embeddings,played_dev_card_masks))
        played_dev_card_masks = played_dev_card_masks.squeeze(1).transpose(-1, -2).long()
        played_dev_representations[played_dev_card_masks.repeat(1, 1, played_dev_representations.shape[-1]) == 0] = 0.0
        # played_dev_out = torch.cat((
        #     played_dev_representations.min(dim=1)[0], played_dev_representations.max(dim=1)[0],
        #     played_dev_representations.sum(dim=1)
        # ), dim=-1)
        played_dev_out = played_dev_representations.sum(dim=1)
        played_dev_out = self.relu(self.norm_3(self.proj_played_dev_card(played_dev_out)))

        main_input = self.relu(self.norm_1(self.main_input_layer_1(main_input)))

        final_input = torch.cat((main_input, played_dev_out, hidden_dev_out), dim=-1)
        return self.relu(self.norm_4(self.final_linear_layer(final_input)))


class OtherPlayersModule(nn.Module):
    def __init__(self, main_input_dim, dev_card_embed_dim, dev_card_model_dim, proj_dev_card_dim):
        super(OtherPlayersModule, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.main_input_dim = main_input_dim
        self.dev_card_embed_dim = dev_card_embed_dim
        self.dev_card_model_dim = dev_card_model_dim
        self.proj_dev_card_dim = proj_dev_card_dim

        self.main_input_layer_1 = init_(nn.Linear(main_input_dim, 256))
        self.relu = nn.ReLU()

        self.proj_played_dev_card = init_(nn.Linear(dev_card_model_dim, proj_dev_card_dim))

        self.final_linear_layer = init_(nn.Linear(proj_dev_card_dim + 256, 128))

        self.norm = nn.LayerNorm(dev_card_model_dim)
        self.norm_1 = nn.LayerNorm(256)
        self.norm_2 = nn.LayerNorm(proj_dev_card_dim)
        self.norm_3 = nn.LayerNorm(128)


    def forward(self, main_input, played_dev_cards, dev_card_embedding, played_card_mha):
        if isinstance(played_dev_cards, list):
            played_dev_card_lengths = [len(played_cards) for played_cards in played_dev_cards]
            padded_played_dev_cards = pad_sequence(played_dev_cards, batch_first=True).long()
        else:
            played_dev_card_lengths = (played_dev_cards.shape[-1] - (played_dev_cards == 0).sum(dim=-1)).cpu().data.numpy()
            played_dev_card_lengths[played_dev_card_lengths == 0] = 1
            played_dev_card_lengths = list(played_dev_card_lengths)
            padded_played_dev_cards = played_dev_cards
        played_dev_card_embeddings = dev_card_embedding(padded_played_dev_cards)
        played_dev_card_masks = torch.zeros(main_input.shape[0], 1, 1, padded_played_dev_cards.shape[1],
                                            device=self.dummy_param.device)
        for b in range(main_input.shape[0]):
            played_dev_card_masks[b, ..., :played_dev_card_lengths[b]] = 1.0

        played_dev_representations = self.norm(played_card_mha(played_dev_card_embeddings, played_dev_card_embeddings,
                                                               played_dev_card_embeddings, played_dev_card_masks))
        played_dev_card_masks = played_dev_card_masks.squeeze(1).transpose(-1, -2).long()
        played_dev_representations[played_dev_card_masks.repeat(1, 1, played_dev_representations.shape[-1]) == 0] = 0.0
        # played_dev_out = torch.cat((
        #     played_dev_representations.min(dim=1)[0], played_dev_representations.max(dim=1)[0],
        #     played_dev_representations.sum(dim=1)
        # ), dim=-1)
        played_dev_out = played_dev_representations.sum(dim=1)
        played_dev_out = self.relu(self.norm_2(self.proj_played_dev_card(played_dev_out)))

        main_input = self.relu(self.norm_1(self.main_input_layer_1(main_input)))

        final_input = torch.cat((main_input, played_dev_out), dim=-1)
        return self.relu(self.norm_3(self.final_linear_layer(final_input)))