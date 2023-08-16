# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F

from llama.tokenizer import Tokenizer
from llama.model_with_past import Transformer
from llama.logits_processor import top_k_top_p_filtering


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: List[str],
        stop_tokens: str = '\n\n',
        max_gen_len: int = 128,
        temperature: float = 0,
        top_p: float = 1.0,
        top_k: float = 0,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        # total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)
        total_len = max_gen_len + max_prompt_size

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        all_finish = [False for beam in range(bsz)] 
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if top_p < 1.0 or top_k > 0:
                log_probs = F.log_softmax(logits / temperature, dim=-1)
                # next_token = sample_top_p(probs, top_p)
                log_probs = top_k_top_p_filtering(log_probs, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(torch.exp(log_probs), num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

            for beam_id, beam in enumerate(tokens.tolist()):
                t = beam[len(prompt_tokens[beam_id]): len(prompt_tokens[beam_id]) + max_gen_len]
                # cut to eos tok if any
                try:
                    t = t[: t.index(self.tokenizer.pad_id)]
                except ValueError:
                    pass
                if self.tokenizer.decode(t).endswith(stop_tokens):
                    all_finish[beam_id] = True

            if all(all_finish):
                break

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[len(prompt_tokens[i]): len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.pad_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t).split(stop_tokens)[0])
        return decoded

    def generate_with_past(
        self,
        prompts: List[str],
        stop_tokens: str = '\n\n',
        max_gen_len: int = 128,
        temperature: float = 0,
        top_p: float = 1.0,
        top_k: float = 0,
        past_key_values: Optional[List[torch.FloatTensor]] = None, 
        prev_pos: int = 0,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        # assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        if max_prompt_size >= params.max_seq_len:
            return ["" for seq in range(bsz)], past_key_values, prev_pos

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)
        # total_len = max_gen_len + max_prompt_size

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        # prev_pos = 0
        all_finish = [False for beam in range(bsz)] 
        # past_key_values = None
        min_prev_pos = None
        min_past = None
        for cur_pos in range(start_pos, total_len):
            # print('generate token at:', cur_pos)
            logits, past_key_values = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos, past_key_values)
            if top_p < 1.0 or top_k > 0 or temperature < 1.0:
                log_probs = F.log_softmax(logits / temperature, dim=-1)
                # next_token = sample_top_p(probs, top_p)
                if top_p < 1.0 or top_k > 0:
                    log_probs = top_k_top_p_filtering(log_probs, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(torch.exp(log_probs), num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

            for beam_id, beam in enumerate(tokens.tolist()):
                t = beam[len(prompt_tokens[beam_id]): len(prompt_tokens[beam_id]) + max_gen_len]
                # cut to eos tok if any
                try:
                    t = t[: t.index(self.tokenizer.pad_id)]
                except ValueError:
                    pass
                if self.tokenizer.decode(t).endswith(stop_tokens):
                    all_finish[beam_id] = True
                    if min_prev_pos is None:
                        min_prev_pos = cur_pos
                        min_past = past_key_values

            if all(all_finish):
                break

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[len(prompt_tokens[i]): len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.pad_id)]
            except ValueError:
                pass
            try:
                decoded.append(self.tokenizer.decode(t).split(stop_tokens)[0])
            except:
                print(t)
                print(self.tokenizer.n_words)
        return decoded, min_past, min_prev_pos


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
