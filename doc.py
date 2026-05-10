import os
import json
import torch
import torch.utils.data.dataset

from typing import Optional, List

from config import args
from triplet import reverse_triplet
from triplet_mask import construct_mask, construct_self_negative_mask
from dict_hub import get_entity_dict, get_link_graph, get_tokenizer
from logger_config import logger

entity_dict = get_entity_dict()
if args.use_link_graph:
    get_link_graph()


def _custom_tokenize(text: str, text_pair: Optional[str] = None) -> dict:
    tokenizer = get_tokenizer()
    if text_pair:
        combined = f"{text} [REL] {text_pair}"
    else:
        combined = text
    encoded_inputs = tokenizer(
        text=combined,
        add_special_tokens=True,
        max_length=args.max_num_tokens,
        truncation=True,
        return_attention_mask=True,
    )
    return encoded_inputs


def _parse_entity_name(entity: str) -> str:
    if args.task.lower() == 'wn18rr':
        entity = ' '.join(entity.split('_')[:-2])
        return entity
    return entity or ''


def _concat_name_desc(entity: str, entity_desc: str) -> str:
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()
    if entity_desc:
        return '{}: {}'.format(entity, entity_desc)
    return entity


def get_neighbor_desc(head_id: str, tail_id: str = None) -> str:
    neighbor_ids = get_link_graph().get_neighbor_ids(head_id)
    if not args.is_test:
        neighbor_ids = [n_id for n_id in neighbor_ids if n_id != tail_id]
    entities = [entity_dict.get_entity_by_id(n_id).entity for n_id in neighbor_ids]
    entities = [_parse_entity_name(entity) for entity in entities]
    return ' '.join(entities)


def _is_medical_kgc_task(task: str) -> bool:
    t = (task or '').lower().replace('-', '_')
    return t in ('primekg', 'tcm_kg', 'tcmkg')


def _should_append_neighbor_desc(entity_desc: str) -> bool:
    """Whether to append 1-hop neighbor *names* from the training link graph.

    The legacy rule ``len(desc.split()) < 20`` skips most long English blurbs
    (common in PrimeKG), so those entities never receive graph context. For
    Chinese text without spaces, ``split()`` is almost always length 1, so
    the old rule was fine; we combine character-based and word-based heuristics
    and relax thresholds on medical KGC tasks.
    """
    s = (entity_desc or '').strip()
    if not s:
        return True
    n_cjk = sum(1 for c in s if '\u4e00' <= c <= '\u9fff')
    n_words = len(s.split())
    n_chars = len(s)
    if n_cjk >= 6:
        return n_cjk < 260
    if _is_medical_kgc_task(args.task):
        return n_words < 48 or n_chars < 720
    return n_words < 22 or n_chars < 360


class Example:

    def __init__(self, head_id, relation, tail_id, **kwargs):
        self.head_id = head_id
        self.tail_id = tail_id
        self.relation = relation

    @property
    def head_desc(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity_desc

    @property
    def tail_desc(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity_desc

    @property
    def head(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity

    @property
    def tail(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity

    def vectorize(self) -> dict:
        head_desc, tail_desc = self.head_desc, self.tail_desc
        if args.use_link_graph:
            if _should_append_neighbor_desc(head_desc):
                head_desc += ' ' + get_neighbor_desc(head_id=self.head_id, tail_id=self.tail_id)
            if _should_append_neighbor_desc(tail_desc):
                tail_desc += ' ' + get_neighbor_desc(head_id=self.tail_id, tail_id=self.head_id)

        head_word = _parse_entity_name(self.head)
        head_text = _concat_name_desc(head_word, head_desc)
        hr_encoded = _custom_tokenize(text=head_text, text_pair=self.relation)

        head_encoded = _custom_tokenize(text=head_text)

        tail_word = _parse_entity_name(self.tail)
        tail_encoded = _custom_tokenize(text=_concat_name_desc(tail_word, tail_desc))

        return {'hr_token_ids': hr_encoded['input_ids'],
                'hr_mask': hr_encoded['attention_mask'],
                'tail_token_ids': tail_encoded['input_ids'],
                'tail_mask': tail_encoded['attention_mask'],
                'head_token_ids': head_encoded['input_ids'],
                'head_mask': head_encoded['attention_mask'],
                'obj': self}


class Dataset(torch.utils.data.dataset.Dataset):

    def __init__(self, path, task, examples=None):
        self.path_list = path.split(',') if path else []
        self.task = task
        if examples:
            self.examples = examples
        else:
            self.examples = []
            for path in self.path_list:
                if not self.examples:
                    self.examples = load_data(path)
                else:
                    self.examples.extend(load_data(path))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index].vectorize()


def load_data(path: str,
              add_forward_triplet: bool = True,
              add_backward_triplet: bool = True) -> List[Example]:
    assert path.endswith('.json'), 'Unsupported format: {}'.format(path)
    assert add_forward_triplet or add_backward_triplet
    logger.info('In test mode: {}'.format(args.is_test))

    data = json.load(open(path, 'r', encoding='utf-8'))
    logger.info('Load {} examples from {}'.format(len(data), path))

    cnt = len(data)
    examples = []
    for i in range(cnt):
        obj = data[i]
        if add_forward_triplet:
            examples.append(Example(**obj))
        if add_backward_triplet:
            examples.append(Example(**reverse_triplet(obj)))
        data[i] = None

    return examples


def collate(batch_data: List[dict]) -> dict:
    tokenizer = get_tokenizer()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    hr_token_ids, hr_mask = to_indices_and_mask(
        [torch.LongTensor(ex['hr_token_ids']) for ex in batch_data],
        pad_token_id=pad_id)
    hr_attn_mask = to_indices_and_mask(
        [torch.LongTensor(ex['hr_mask']) for ex in batch_data],
        need_mask=False, pad_token_id=0)

    tail_token_ids, tail_mask = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_ids']) for ex in batch_data],
        pad_token_id=pad_id)
    tail_attn_mask = to_indices_and_mask(
        [torch.LongTensor(ex['tail_mask']) for ex in batch_data],
        need_mask=False, pad_token_id=0)

    head_token_ids, head_mask = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_ids']) for ex in batch_data],
        pad_token_id=pad_id)
    head_attn_mask = to_indices_and_mask(
        [torch.LongTensor(ex['head_mask']) for ex in batch_data],
        need_mask=False, pad_token_id=0)

    batch_exs = [ex['obj'] for ex in batch_data]
    batch_dict = {
        'hr_token_ids': hr_token_ids,
        'hr_mask': hr_attn_mask,
        'tail_token_ids': tail_token_ids,
        'tail_mask': tail_attn_mask,
        'head_token_ids': head_token_ids,
        'head_mask': head_attn_mask,
        'batch_data': batch_exs,
        'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,
        'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
    }

    return batch_dict


def to_indices_and_mask(batch_tensor, pad_token_id=0, need_mask=True):
    mx_len = max([t.size(0) for t in batch_tensor])
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, mx_len).fill_(pad_token_id)
    if need_mask:
        mask = torch.ByteTensor(batch_size, mx_len).fill_(0)
    for i, t in enumerate(batch_tensor):
        indices[i, :len(t)].copy_(t)
        if need_mask:
            mask[i, :len(t)].fill_(1)
    if need_mask:
        return indices, mask
    else:
        return indices
