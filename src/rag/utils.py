


import torch
import numpy as np
from tqdm import tqdm, trange
from typing import Any, List, Union, Tuple, Optional
from FlagEmbedding import FlagReranker


def _patched_compute_score_single_gpu(self, sentence_pairs, batch_size=None,
                                       query_max_length=None, max_length=None,
                                       normalize=None, device=None, **kwargs):
    if batch_size is None: batch_size = self.batch_size
    if max_length is None: max_length = self.max_length
    if query_max_length is None:
        query_max_length = self.query_max_length if self.query_max_length else max_length * 3 // 4
    if normalize is None: normalize = self.normalize
    if device is None: device = self.target_devices[0]

    if device == "cpu": self.use_fp16 = False
    if self.use_fp16: self.model.half()
    self.model.to(device)
    self.model.eval()

    assert isinstance(sentence_pairs, list)
    if isinstance(sentence_pairs[0], str):
        sentence_pairs = [sentence_pairs]

    tok = self.tokenizer
    cls_id = tok.cls_token_id
    sep_id = tok.sep_token_id

    all_inputs = []
    for start_index in trange(0, len(sentence_pairs), batch_size, desc="pre tokenize",
                               disable=len(sentence_pairs) < batch_size):
        sentences_batch = sentence_pairs[start_index:start_index + batch_size]
        queries = [s[0] for s in sentences_batch]
        passages = [s[1] for s in sentences_batch]

        queries_inputs_batch = tok(
            queries, return_tensors=None, add_special_tokens=False,
            max_length=query_max_length, truncation=True
        )['input_ids']
        passages_inputs_batch = tok(
            passages, return_tensors=None, add_special_tokens=False,
            max_length=max_length, truncation=True
        )['input_ids']

        for q_inp, d_inp in zip(queries_inputs_batch, passages_inputs_batch):
            # Manually do what prepare_for_model does: [CLS] query [SEP] passage [SEP]
            input_ids = [cls_id] + q_inp + [sep_id] + d_inp + [sep_id]
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length - 1] + [sep_id]
            attention_mask = [1] * len(input_ids)
            all_inputs.append({'input_ids': input_ids, 'attention_mask': attention_mask})

    # Sort by length for less padding
    length_sorted_idx = np.argsort([-len(x['input_ids']) for x in all_inputs])
    all_inputs_sorted = [all_inputs[i] for i in length_sorted_idx]

    def sigmoid(x):
        return float(1 / (1 + np.exp(-x)))

    flag = False
    while not flag:
        try:
            test_batch = tok.pad(
                all_inputs_sorted[:min(len(all_inputs_sorted), batch_size)],
                padding=True, return_tensors='pt'
            ).to(device)
            self.model(**test_batch, return_dict=True).logits.view(-1,).float()
            flag = True
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            batch_size = batch_size * 3 // 4

    all_scores = []
    for start_index in tqdm(range(0, len(all_inputs_sorted), batch_size),
                            desc="Compute Scores", disable=len(all_inputs_sorted) < batch_size):
        batch = all_inputs_sorted[start_index:start_index + batch_size]
        inputs = tok.pad(batch, padding=True, return_tensors='pt').to(device)
        scores = self.model(**inputs, return_dict=True).logits.view(-1,).float()
        all_scores.extend(scores.cpu().detach().numpy().tolist())

    all_scores = [all_scores[idx] for idx in np.argsort(length_sorted_idx)]
    if normalize:
        all_scores = [sigmoid(score) for score in all_scores]
    return all_scores



def init_reranker():
   from FlagEmbedding import FlagReranker
   from FlagEmbedding.inference.reranker.encoder_only.base import BaseReranker
   BaseReranker.compute_score_single_gpu = _patched_compute_score_single_gpu
    
   reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
   return reranker
