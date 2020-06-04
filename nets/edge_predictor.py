import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
from utils.tensor_functions import compute_in_batches

from nets.graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many
from nets.attention_model import AttentionModel


class EdgeAttentionModel(AttentionModel):

    def __init__(self, **kwargs):
        super(EdgeAttentionModel, self).__init__(**kwargs)
        self.project_from_heads = nn.Linear(self.n_heads, 1, bias=False)

    def forward(self, input, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """

        if self.checkpoint_encoder and self.training:  # Only checkpoint if we need gradients
            embeddings, _ = checkpoint(self.embedder, self._init_embed(input))
        else:
            embeddings, _ = self.embedder(self._init_embed(input))

        _log_p, actions, adjacency = self._inner(input, embeddings)

        cost, mask = self.problem.get_costs(input, adjacency)
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(_log_p, actions, mask)
        if return_pi:
            return cost, ll, pi

        return cost, ll


    def _inner(self, input, embeddings):

        probs = []
        actions = []
        action_masks = []

        state = self.problem.make_state(input)

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)

        batch_size = state.ids.size(0)

        # Perform decoding steps
        i = 0
        while not (self.shrink_size is None and state.all_finished()):
            log_p, mask = self._get_log_p(fixed, state)

            # Select the next edge
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension

            n_nodes = state.n_nodes
            selected_edge_mask = torch.zeros(batch_size, 1, n_nodes**2)
            selected_edge_mask = selected_edge_mask.scatter(-1, selected[:, None, None], 1)
            selected_edge_mask = selected_edge_mask.view(batch_size, n_nodes, n_nodes)
       
            state = state.update(selected)

            # Collect output of step
            probs.append(log_p[:, 0, :])
            actions.append(selected)
            action_masks.append(selected_edge_mask)

            i += 1

        # Return 1) Probabiliyu distributions 2) Chosen actions (edges) 
        # 3) A combined adjacency matrix
        return torch.stack(probs, 1), torch.stack(actions, 1), torch.sum(torch.stack(action_masks, 1), 1)


    def _get_log_p(self, fixed, state, normalize=True):
        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, _ = self._get_attention_node_data(fixed, state)

        # Compute the mask
        mask = state.get_mask()

        # Compute logits (unnormalized log_p)
        log_p = self._one_to_edge_logits(glimpse_K, glimpse_V, mask)

        assert not torch.isnan(log_p).any()

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()
        return log_p, mask


    def _compute_attention(self, glimpse_Q, glimpse_K, mask):
        n_heads, batch_size, num_steps, embed_dim, _ = glimpse_Q.size()
        key_size = val_size = embed_dim // self.n_heads

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        attention = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1))

        attention = self.project_from_heads(attention.permute(1, 2, 3, 4, 0)).squeeze(-1)

        if mask is not None:
            attention[mask.expand_as(attention)] = -math.inf

        return attention


    def _one_to_edge_logits(self, query, glimpse_K, mask):
        edge_attention = self._compute_attention(query, glimpse_K, mask) 

        assert not torch.isnan(edge_attention).any()

        assert edge_attention.shape[3] == edge_attention.shape[2]
        batch_size, _, n_nodes, n_nodes = edge_attention.shape
        
        # squeeze the probs over edges [n_nodes x n_nodes] in one dimention
        edge_attention = edge_attention.view(batch_size, 1, n_nodes**2)

        assert not torch.isnan(edge_attention).any()

        return edge_attention
