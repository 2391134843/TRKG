import torch

from config import args
from dict_hub import get_link_graph
from triplet import EntityDict

from logger_config import logger

def rerank_by_graph(batch_score, examples, entity_dict: EntityDict):
    if args.neighbor_weight <= 0:
        return

    link_graph = get_link_graph()
    for idx, ex in enumerate(examples):
        neighbor_ids = link_graph.get_n_hop_entity_indices(
            entity_id=ex.tail_id,
            entity_dict=entity_dict,
            n_hop=args.rerank_n_hop)
        if not neighbor_ids:
            continue
        entity_indices = torch.LongTensor(list(neighbor_ids)).to(batch_score.device)
        add_vec = torch.full(
            (len(entity_indices),),
            args.neighbor_weight,
            dtype=batch_score.dtype,
            device=batch_score.device)
        batch_score[idx].index_add_(0, entity_indices, add_vec)
