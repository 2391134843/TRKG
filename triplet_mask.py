import torch

from typing import List

def construct_mask(row_exs, col_exs=None) -> torch.tensor:
    positive_on_diagonal = col_exs is None
    num_row = len(row_exs)
    col_exs = col_exs or row_exs
    num_col = len(col_exs)

    row_entity_ids = torch.LongTensor([ex.tail_id.__hash__() for ex in row_exs])
    col_entity_ids = torch.LongTensor([ex.tail_id.__hash__() for ex in col_exs])
    triplet_mask = (row_entity_ids.unsqueeze(1) != col_entity_ids.unsqueeze(0))
    if positive_on_diagonal:
        triplet_mask.fill_diagonal_(True)
    return triplet_mask

def construct_self_negative_mask(batch_exs) -> torch.tensor:
    batch_size = len(batch_exs)
    mask = torch.ones(batch_size)
    for i, ex in enumerate(batch_exs):
        if ex.head_id == ex.tail_id:
            mask[i] = 0
    return mask.bool()
