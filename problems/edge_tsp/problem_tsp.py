from torch.utils.data import Dataset
import torch
import os
import pickle
from utils.beam_search import beam_search
from problems.tsp.problem_tsp import TSP, TSPDataset
from problems.edge_tsp.state_tsp import StateEdgeTSP


def make_symmetric(adjacency):
    adjacency = adjacency + adjacency.transpose(-2, -1)

    # correct the numbers on the diagonal
    n_nodes = adjacency.shape[-1]
    adjacency[:, torch.eye(n_nodes).bool()] = torch.diagonal(adjacency, dim1=-2, dim2=-1) / 2
    return adjacency


class EdgeTSP(TSP):

    NAME = 'edgetsp'

    @staticmethod
    def get_costs(dataset, adjacency):
        # Check that tours are valid
        # adjacency contains the mask of selected edges
        assert adjacency.shape[-1] == adjacency.shape[-2]

        unique_nodes = adjacency.sum(-1) > 0
        n_unique = unique_nodes.sum(-1)
        is_valid_tour = unique_nodes.all()
        
        #assert is_valid_tour, "Invalid tour"

        distances = (dataset[:, :, None, :] - dataset[:, None, :, :]).norm(p=2, dim=-1)
        # costs of all added edges
        costs = torch.sum(distances * adjacency, (-2, -1))
        
        n_nodes = adjacency.size(-1)
        n_missing_nodes = n_nodes - n_unique
        
        costs += n_missing_nodes * 10 # penalize each path by the number of missing nodes

        symmetric_adj = make_symmetric(adjacency)
        node_degrees = adjacency.sum(-1)

        # Each node in TSP path has to have degree of 2
        n_incorrect_degree = (node_degrees != 2).sum(-1)
        costs += n_incorrect_degree * 5 # penalize each node for incorrect degree

        # todo: prevent the paths from being two disconnected loops

        return costs, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateEdgeTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = EdgeTSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)