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
            # todo: remake selection function
            selected = self._select_node(log_p.exp().squeeze(1), mask.squeeze(1))  # Squeeze out steps dimension

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

        # Return 1) Probability distributions 2) Chosen actions (edges) 
        # 3) A combined adjacency matrix
        return torch.stack(probs, 1), torch.stack(actions, 1), torch.sum(torch.stack(action_masks, 1), 1)


    def _get_log_p(self, fixed, state, normalize=True):
        # Compute keys and values for the nodes
        glimpse_Q, glimpse_K, glimpse_V = self._get_attention_node_data(fixed, state)

        # Compute the mask
        mask = state.get_mask()

        # Compute logits (unnormalized log_p)
        log_p = self._one_to_edge_logits(glimpse_Q, glimpse_K, glimpse_V,
            mask, fixed.context_node_projected)

        assert not torch.isnan(log_p).any()

        # todo: normalization
        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()
        return log_p, mask


    def _one_to_edge_logits(self, glimpse_Q, glimpse_K, glimpse_V, mask, graph_embedding):
        # TODO: use different projections for two types of attention 
        # todo: do we want to use a mask in _compute_parwise_attention x?
        #node_attention = self._compute_parwise_attention(glimpse_Q, glimpse_K) 

        node_connectivity_glimpse = self._compute_graph_glimpse(
            glimpse_V, glimpse_Q, glimpse_K, mask, graph_embedding)
        node_connectivity_glimpse = node_connectivity_glimpse.unsqueeze(0)

        # todo: correct normalization
        # for_choosing_src = self.project  (node_connectivity_glimpse)

        # # todo: correct normalization
        # for_choosing_dst = self.project (node_connectivity_glimpse)


        # connectivity_attention = self._compute_parwise_attention(
        #     for_choosing_dst, for_choosing_dst) 

        connectivity_attention = self._compute_parwise_attention(
            node_connectivity_glimpse, node_connectivity_glimpse, mask) 
        
        # edge_attention = self._compute_final_attention_mask(
        #     [node_attention, connectivity_attention], mask)

        # todo: correct normalization
        edge_attention = self._compute_final_attention_mask(
            connectivity_attention, mask)

        assert not torch.isnan(edge_attention).any()

        assert edge_attention.shape[3] == edge_attention.shape[2]
        batch_size, _, n_nodes, n_nodes = edge_attention.shape
        
        # squeeze the probs over edges [n_nodes x n_nodes] in one dimention
        edge_attention = edge_attention.view(batch_size, 1, n_nodes**2)

        assert not torch.isnan(edge_attention).any()

        # return for_choosing_src, edge_attention
        return edge_attention


    def _compute_final_attention_mask(self, attention, mask):
        if isinstance(attention, list):
            attention = torch.cat(attention, -1)

        attention = self.project_attention(attention).squeeze(-1)

        if mask is not None:
            attention[mask.expand_as(attention)] = -math.inf

        return attention




    def _compute_parwise_attention(self, glimpse_Q, glimpse_K, mask):
        n_heads, batch_size, num_steps, n_nodes, embed_dim = glimpse_Q.size()
        key_size = val_size = embed_dim // self.n_heads

        # Dot product attention
        # attention = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1))
        # attention = self.project_from_heads(attention.permute(1, 2, 3, 4, 0)).squeeze(-1)

        # todo: do some projection here?
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

