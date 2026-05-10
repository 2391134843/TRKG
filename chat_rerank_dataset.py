"""
Chat Reranking Dataset: formats TRKG candidates as multiple-choice chat prompts
for GRPO training with Qwen2.5-Instruct.
"""
import json
import random
import torch
import torch.utils.data

from typing import List, Dict, Optional
from logger_config import logger

OPTION_LABELS = (
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("abcdefghijklmnopqrstuvwxyz")
    + [str(i) for i in range(10)]
    + [chr(c) for c in range(0x3B1, 0x3B1 + 23)]
    + [chr(c) for c in range(0x391, 0x391 + 23)]
)
assert len(OPTION_LABELS) >= 100, \
    f"expected >=100 single-token labels, got {len(OPTION_LABELS)}"

def _parse_entity_name(entity: str, task: str) -> str:
    if task.lower() == 'wn18rr':
        return ' '.join(entity.split('_')[:-2])
    return entity or ''

def build_chat_prompt(head: str, head_desc: str, relation: str,
                      candidates: List[Dict], task: str) -> str:
    """Build a multiple-choice chat prompt for KGC reranking."""
    head_name = _parse_entity_name(head, task)
    head_text = head_name
    if head_desc and not head_desc.startswith(head_name):
        head_text = f"{head_name}: {head_desc}"
    elif head_desc:
        head_text = head_desc

    def _short_desc(ent_name, ent_desc, max_words=15):
        name = _parse_entity_name(ent_name, task)
        if ent_desc and not ent_desc.startswith(name):
            desc = f"{name}: {' '.join(ent_desc.split()[:max_words])}"
        elif ent_desc:
            desc = ' '.join(ent_desc.split()[:max_words])
        else:
            desc = name
        return desc

    options_text = '\n'.join(
        f"{OPTION_LABELS[i]}. {_short_desc(c['entity'], c.get('entity_desc', ''))}"
        for i, c in enumerate(candidates)
    )

    prompt = (
        f"Given the following knowledge graph query, select the most likely tail entity.\n\n"
        f"Head Entity: {head_text}\n"
        f"Relation: {relation}\n\n"
        f"Candidates:\n{options_text}\n\n"
        f"Answer with just the letter."
    )
    return prompt

class ChatRerankDataset(torch.utils.data.Dataset):
    """
    Dataset that loads pre-generated candidates and formats them as chat prompts.
    Each item returns:
      - messages: list of chat messages (system + user)
      - correct_idx: index of the correct option (0-based)
      - num_options: number of candidate options
      - meta: original example metadata
    """

    def __init__(self, candidates_path: str, task: str,
                 max_candidates: int = 10, shuffle_candidates: bool = True):
        logger.info(f'Loading candidates from {candidates_path}')
        with open(candidates_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.task = task
        self.max_candidates = max_candidates
        self.shuffle_candidates = shuffle_candidates
        logger.info(f'Loaded {len(self.data)} examples, max_candidates={max_candidates}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        all_cands = item['candidates']

        gt_pos = None
        for i, c in enumerate(all_cands):
            if c['is_correct']:
                gt_pos = i
                break

        candidates = all_cands[:self.max_candidates]
        if gt_pos is not None and gt_pos >= self.max_candidates:
            candidates[-1] = all_cands[gt_pos]

        if self.shuffle_candidates:
            random.shuffle(candidates)

        correct_idx = -1
        for i, c in enumerate(candidates):
            if c['is_correct']:
                correct_idx = i
                break
        assert correct_idx >= 0, f"Ground truth not found in candidates for {item['head_id']}"

        prompt = build_chat_prompt(
            head=item['head'],
            head_desc=item.get('head_desc', ''),
            relation=item['relation'],
            candidates=candidates,
            task=self.task,
        )

        messages = [
            {"role": "system", "content": "You are a knowledge graph expert. Answer with a single letter."},
            {"role": "user", "content": prompt},
        ]

        return {
            'messages': messages,
            'correct_idx': correct_idx,
            'num_options': len(candidates),
            'meta': {
                'head': item['head'],
                'relation': item['relation'],
                'tail': item['tail'],
                'direction': item['direction'],
            }
        }

def collate_chat(batch: List[dict], tokenizer, max_length: int = 512) -> dict:
    """
    Tokenize a batch of chat examples.
    Returns input_ids, attention_mask, correct_idx, option_token_ids, num_options.
    """
    messages_list = [ex['messages'] for ex in batch]
    correct_indices = [ex['correct_idx'] for ex in batch]
    num_options_list = [ex['num_options'] for ex in batch]

    texts = []
    for msgs in messages_list:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        texts.append(text)

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )

    max_opts = max(num_options_list)
    option_token_ids = []
    for label in OPTION_LABELS[:max_opts]:
        tids = tokenizer.encode(label, add_special_tokens=False)
        option_token_ids.append(tids[0])
    option_token_ids = torch.LongTensor(option_token_ids)

    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'correct_idx': torch.LongTensor(correct_indices),
        'option_token_ids': option_token_ids,
        'num_options': torch.LongTensor(num_options_list),
        'meta': [ex['meta'] for ex in batch],
    }
