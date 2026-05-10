import torch

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions.

    Robust to tail batches whose number of in-batch candidates
    (``output.size(1)``) is smaller than ``max(topk)``: in that case
    we right-pad the score matrix with ``-inf`` so ``torch.topk`` is
    always well-defined. Padded indices can never equal the target
    label so they don't poison Acc@k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        if output.size(1) < maxk:
            pad = torch.full(
                (output.size(0), maxk - output.size(1)),
                float('-inf'),
                device=output.device,
                dtype=output.dtype,
            )
            output = torch.cat([output, pad], dim=1)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def ranking_metrics(output, target):
    """Computes Hits@1, Hits@3, Hits@5, Hits@10, MRR, MR from in-batch logits.

    Args:
        output: (batch_size, num_candidates) logit scores
        target: (batch_size,) ground-truth indices

    Returns:
        dict with keys Hits@{1,3,5,10}, MRR, MR (floats, percentage-scaled
        for hits / MRR; MR is the raw mean rank in [1, num_candidates]).
    """
    with torch.no_grad():
        sorted_indices = output.argsort(dim=-1, descending=True)
        target_expanded = target.unsqueeze(-1).expand_as(sorted_indices)
        ranks = (sorted_indices == target_expanded).nonzero(as_tuple=False)[:, 1] + 1

        ranks = ranks.float()
        hit1 = (ranks <= 1).float().mean().item() * 100.0
        hit3 = (ranks <= 3).float().mean().item() * 100.0
        hit5 = (ranks <= 5).float().mean().item() * 100.0
        hit10 = (ranks <= 10).float().mean().item() * 100.0
        mrr = (1.0 / ranks).mean().item() * 100.0
        mr = ranks.mean().item()

        return {'Hits@1': hit1, 'Hits@3': hit3, 'Hits@5': hit5,
                'Hits@10': hit10, 'MRR': mrr, 'MR': mr}
