"""
Neural Collaborative Filtering Model
Based on https://arxiv.org/abs/1708.05031

Author: Ivan Klevanski

"""

import torch
import torch.nn as nn
import torchvision.ops as torch_ops


class NCFNet(nn.Module):
    """
    Neural Collaborative Filtering Model\n
    Based on: https://arxiv.org/abs/1708.05031
    """

    def __init__(self, num_items, num_users, embedding_dim, mlp_layer_sizes=(128, 64)):
        super(NCFNet, self).__init__()

        self.ilv_embedding = nn.Embedding(num_items, embedding_dim)
        self.ulv_embedding = nn.Embedding(num_users, embedding_dim)

        mlp_input_size = embedding_dim * 2  # Concatenation of embeddings fed into the mlp layers
        self.mlp_layers = torch_ops.MLP(mlp_input_size, list(mlp_layer_sizes), inplace=False)
        self.mlp_layers.append(nn.Linear(mlp_layer_sizes[-1], 1))

    def forward(self, item_vec, user_vec):
        concat_ed = torch.concat((self.ilv_embedding(item_vec), self.ulv_embedding(user_vec)), dim=1)
        output = torch.sigmoid(self.mlp_layers(concat_ed))
        return output.view(-1)
