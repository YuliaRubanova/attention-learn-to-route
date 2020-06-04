import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateEdgeTSP(NamedTuple):
    # Fixed input
    loc: torch.Tensor
    dist: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    #first_a: torch.Tensor
    #prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    visited_edges: torch.Tensor
    lengths: torch.Tensor
    #cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step
    n_nodes: int # number of nodes

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                visited_=self.visited_[key],
                visited_edges=self.visited_edges[key],
                lengths=self.lengths[key],
            )
        return super(StateEdgeTSP, self).__getitem__(key)

    @staticmethod
    def initialize(loc, visited_dtype=torch.uint8):
        batch_size, n_loc, _ = loc.size()

        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)
        return StateEdgeTSP(
            loc=loc,
            dist=(loc[:, :, None, :] - loc[:, None, :, :]).norm(p=2, dim=-1),
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, n_loc,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            visited_edges=(
                torch.zeros(
                    batch_size, 1, n_loc, n_loc,
                    dtype=torch.uint8, device=loc.device
                )
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps
            n_nodes=n_loc,
        )

    # todo:
    def get_final_cost(self):
        raise Exception("Not implemented")

        # assert self.all_finished()
        # return self.lengths + (self.loc[self.ids, self.first_a, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected_edge_ids):
        # each element in "selected" is a chosen edge represented as binary [n_nodes x n_nodes] matrix
        visited_edges = self.visited_edges

        batch_size, _, n_nodes, n_nodes = visited_edges.shape
        visited_edges = visited_edges.view(batch_size, 1, n_nodes**2)
        visited_edges = visited_edges.scatter(-1, selected_edge_ids[:, None, None], 1)
        visited_edges = visited_edges.view(batch_size, 1, n_nodes, n_nodes)

        # To get a mask over visited nodes, just note which nodes have outgoing edges
        visited_ = visited_edges.sum(-1) > 0

        # Compute the total distace of the route
        lengths = torch.sum(self.dist * visited_edges.squeeze(1), (-2, -1))

        return self._replace(visited_=visited_, visited_edges=visited_edges,
                             lengths=lengths, i=self.i + 1)

    def all_finished(self):
        # Exactly n steps
        return self.i.item() >= self.loc.size(-2)

    def get_mask(self):
        visited_edges = self.visited_edges
        
        # Since we are deadling with undirected graphs, flip the matrix to make it symmetric
        visited_edges = visited_edges + visited_edges.transpose(-2, -1)

        return visited_edges > 0

    def get_nn(self, k=None):
        # Insert step dimension
        # Nodes already visited get inf so they do not make it
        if k is None:
            k = self.loc.size(-2) - self.i.item()  # Number of remaining
        return (self.dist[self.ids, :, :] + self.visited.float()[:, :, None, :] * 1e6).topk(k, dim=-1, largest=False)[1]

    # todo
    def get_nn_current(self, k=None):
        raise Exception("Not implemented")

        # assert False, "Currently not implemented, look into which neighbours to use in step 0?"
        # # Note: if this is called in step 0, it will have k nearest neighbours to node 0, which may not be desired
        # # so it is probably better to use k = None in the first iteration
        # if k is None:
        #     k = self.loc.size(-2)
        # k = min(k, self.loc.size(-2) - self.i.item())  # Number of remaining
        # return (
        #     self.dist[
        #         self.ids,
        #         self.prev_a
        #     ] +
        #     self.visited.float() * 1e6
        # ).topk(k, dim=-1, largest=False)[1]

    def construct_solutions(self, actions):
        return actions
