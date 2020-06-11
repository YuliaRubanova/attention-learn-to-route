import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
import numpy as np
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

        self.project_for_src_node = nn.Linear(self.embedding_dim, 1)
        self.project_for_dst_node = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.project_attention = nn.Linear(self.embedding_dim * 2 +1, 1, bias=False)
        self.project_compatibility = nn.Linear(self.embedding_dim // self.n_heads * 3 + 1, 1, bias=False)

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

        node_log_p, edge_log_p, node_actions, edge_actions, adjacency = self._inner(input, embeddings)

        cost, mask = self.problem.get_costs(input, adjacency)

        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths

        node_ll = self._calc_log_likelihood(node_log_p, node_actions, None)
        edge_ll = self._calc_log_likelihood(edge_log_p, edge_actions, None)

        ll = node_ll + edge_ll

        if return_pi:
            return cost, ll, adjacency

        return cost, ll


    def _inner(self, input, embeddings):
        node_probs = []
        edge_probs = []

        node_actions = []
        edge_actions = []

        edge_masks = []

        state = self.problem.make_state(input)

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)

        batch_size = state.ids.size(0)

        # Perform decoding steps
        i = 0
        while not (self.shrink_size is None and state.all_finished()):
            log_p_nodes, log_p_edges, mask = self._get_log_p(fixed, state)

            # Select the source node
            selected_node = self._select_node(log_p_nodes.exp().squeeze(1), None)  # Squeeze out steps dimension

            batch_size, n_steps, n_nodes, n_nodes = log_p_edges.shape

            selected_node_to_gather = selected_node[:, None, None, None].expand(batch_size, n_steps, 1, n_nodes)

            selected_dst_logits = log_p_edges.gather(2, selected_node_to_gather).squeeze(1)

            # Check that each row is normalized to sum to 1
            assert (selected_dst_logits.exp().sum(-1) - 1 < 1e-6).all()

            selected_dst_nodes = self._select_node(selected_dst_logits.exp().squeeze(1), None)

            n_nodes = state.n_nodes
            selected_edge_mask = torch.zeros(batch_size, 1, n_nodes**2)
            index = selected_node * n_nodes + selected_dst_nodes
            selected_edge_mask = selected_edge_mask.scatter(-1, index[:, None, None], 1)
            selected_edge_mask = selected_edge_mask.view(batch_size, n_nodes, n_nodes)

            # for mask_, x, y in zip(selected_edge_mask, selected_node, selected_dst_nodes):
            #     mask_[x, y] = 1

            assert selected_edge_mask[0, selected_node[0], selected_dst_nodes[0]] == 1

            assert (selected_edge_mask.sum(-1).sum(-1) - 1 < 1e-6).all()

            state = state.update(selected_node, selected_dst_nodes)

            # Collect output of step
            node_probs.append(log_p_nodes[:, 0, :])
            edge_probs.append(selected_dst_logits[:, 0, :])

            node_actions.append(selected_node)
            edge_actions.append(selected_dst_nodes)

            edge_masks.append(selected_edge_mask)

            i += 1

        # Return 1) Probability distributions 2) Chosen actions (edges) 
        # 3) A combined adjacency matrix
        return torch.stack(node_probs, 1), torch.stack(edge_probs, 1), \
            torch.stack(node_actions, 1), torch.stack(edge_actions, 1), \
            torch.sum(torch.stack(edge_masks, 1), 1)


    def _get_log_p(self, fixed, state, normalize=True):
        # Compute keys and values for the nodes
        glimpse_Q, glimpse_K, glimpse_V = self._get_attention_node_data(fixed, state)

        # Compute the mask
        mask = state.get_mask()

        # Compute logits (unnormalized log_p)
        log_p_nodes, log_p_edges = self._one_to_edge_logits(glimpse_Q, glimpse_K, glimpse_V,
            mask, fixed.context_node_projected)

        assert not torch.isnan(log_p_nodes).any()
        assert not torch.isnan(log_p_edges).any()

        if normalize:
            log_p_nodes = torch.log_softmax(log_p_nodes / self.temp, dim=-1)
            log_p_edges = torch.log_softmax(log_p_edges / self.temp, dim=-1)

        assert not torch.isnan(log_p_nodes).any()
        assert not torch.isnan(log_p_edges).any()

        return log_p_nodes, log_p_edges, mask


    def _one_to_edge_logits(self, glimpse_Q, glimpse_K, glimpse_V, mask, graph_embedding):
        node_connectivity_glimpse = self._compute_graph_glimpse(
            glimpse_V, glimpse_Q, glimpse_K, mask, graph_embedding)
        node_connectivity_glimpse = node_connectivity_glimpse.unsqueeze(0)

        node_selection_logits = self.project_for_src_node(node_connectivity_glimpse).squeeze(-1)
        node_selection_logits = node_selection_logits.squeeze(0) # squeeze heads dimension
        
        dst_nodes = self.project_for_dst_node(node_connectivity_glimpse)

        edge_attention_input = self._compute_parwise_attention_input(
            dst_nodes, dst_nodes, mask) 

        edge_attention = self._compute_final_attention_mask(
            edge_attention_input, mask)

        assert not torch.isnan(edge_attention).any()

        assert edge_attention.shape[3] == edge_attention.shape[2]
        batch_size, _, n_nodes, n_nodes = edge_attention.shape
        
        # squeeze the probs over edges [n_nodes x n_nodes] in one dimention
        #edge_attention = edge_attention.view(batch_size, 1, n_nodes**2)

        assert not torch.isnan(edge_attention).any()
        return node_selection_logits, edge_attention


    def _compute_final_attention_mask(self, attention, mask):
        if isinstance(attention, list):
            attention = torch.cat(attention, -1)

        attention = self.project_attention(attention).squeeze(-1)

        if mask is not None:
            attention[mask.expand_as(attention)] = -math.inf

        return attention




    def _compute_parwise_attention_input(self, glimpse_Q, glimpse_K, mask):
        n_heads, batch_size, num_steps, n_nodes, embed_dim = glimpse_Q.size()
        key_size = val_size = embed_dim // self.n_heads

        # Dot product attention
        # attention = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1))
        # attention = self.project_from_heads(attention.permute(1, 2, 3, 4, 0)).squeeze(-1)

        expanded_shape = (n_heads, batch_size, num_steps, n_nodes, n_nodes, embed_dim)

        attention = torch.cat((
            glimpse_Q.unsqueeze(3).expand(expanded_shape),
            glimpse_K.unsqueeze(4).expand(expanded_shape)
            ), 
        -1)

        attention = attention.permute(1, 2, 3, 4, 5, 0)

        attention = attention.reshape(batch_size, num_steps, 
            n_nodes, n_nodes, -1)

        attention = torch.cat((attention, mask[:,:,:,:,None].float()), -1)
        return attention

 
    def _compute_graph_glimpse(self, query, glimpse_K, glimpse_V, mask, graph_embedding):
        # Compute glimpse only on the graph that is not included into the path yet
        batch_size, num_steps, n_nodes, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, n_nodes, self.n_heads, key_size).permute(3, 0, 1, 2, 4)
        graph_embedding = graph_embedding.view(batch_size, num_steps, self.n_heads, key_size).permute(2, 0, 1, 3)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        #compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))

        expanded_shape = (self.n_heads, batch_size, num_steps, n_nodes, n_nodes, key_size)

        compatibility = torch.cat((
            glimpse_Q.unsqueeze(3).expand(expanded_shape),
            glimpse_K.unsqueeze(4).expand(expanded_shape),
            mask[None,:,:,:,:,None].expand((*expanded_shape[:-1], 1)).float(),
            graph_embedding[:,:,:,None,None,:].expand(expanded_shape),
            ), 
        -1)

        compatibility = self.project_compatibility(compatibility).squeeze(-1)

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(batch_size, num_steps, n_nodes, self.n_heads * val_size))

        return glimpse.squeeze(-2)

