# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

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
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        # assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

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
        past_key_values = None
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

class Contrastive_LLaMA:
    def __init__(self, model: Transformer, contrastive_model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.contrastive_model = contrastive_model 
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: List[str],
        contrastive_prompts: List[str],
        stop_tokens: str = '\n\n',
        answer_prefix: str = 'answer is',
        max_gen_len: int = 128,
        temperature: float = 0,
        top_p: float = 0,
        top_k: int = 0,
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

        contrastive_prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in contrastive_prompts]
        contrastive_tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(contrastive_prompt_tokens):
            contrastive_tokens[k, : len(t)] = torch.tensor(t).long()
        contrastive_input_text_mask = contrastive_tokens != self.tokenizer.pad_id
        contrastive_start_pos = len(contrastive_prompt_tokens[0])
        contrastive_cur_pos = contrastive_start_pos
        contrastive_prev_pos = 0
        do_contrastive = False 

        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

            log_prob1 = F.log_softmax(logits / temperature, dim=-1)
            log_prob1 = top_k_top_p_filtering(log_prob1, top_k=top_k, top_p=top_p)

            if do_contrastive:
                contrastive_logits = self.contrastive_model.forward(contrastive_tokens[:, contrastive_prev_pos:contrastive_cur_pos], contrastive_prev_pos)
                log_prob2 = F.log_softmax(contrastive_logits, dim=-1)

                log_prob_diff = log_prob1 - log_prob2

                next_token = torch.argmax(log_prob_diff, dim=-1)
            else:
                next_token = torch.argmax(log_prob1, dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

            if do_contrastive:
                contrastive_tokens[:, contrastive_cur_pos] = next_token
                contrastive_prev_pos = contrastive_cur_pos
                contrastive_cur_pos += 1

            t = tokens.tolist()[0][len(prompt_tokens[0]): len(prompt_tokens[0]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.pad_id)]
            except ValueError:
                pass
            if self.tokenizer.decode(t).endswith(stop_tokens):
                break
            if self.tokenizer.decode(t).endswith(answer_prefix):
                do_contrastive = True
            # if do_contrastive and self.tokenizer.decode(t).endswith('.'):
            #     contrastive_tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
            #     for k, t in enumerate(contrastive_prompt_tokens):
            #         contrastive_tokens[k, : len(t)] = torch.tensor(t).long()
            #     contrastive_prev_pos = 0
            #     contrastive_cur_pos = contrastive_start_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[len(prompt_tokens[i]): len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.pad_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

class Sentence_Contrastive_LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: List[str],
        contrastive_prompts: List[str],
        stop_tokens: str = '\n\n',
        answer_prefix: str = 'answer is',
        max_gen_len: int = 128,
        temperature: float = 0,
        top_p: float = 0,
        top_k: int = 0,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        # assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

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

        contrastive_prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in contrastive_prompts]
        contrastive_tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(contrastive_prompt_tokens):
            contrastive_tokens[k, : len(t)] = torch.tensor(t).long()
        contrastive_input_text_mask = contrastive_tokens != self.tokenizer.pad_id
        contrastive_start_pos = len(contrastive_prompt_tokens[0])
        contrastive_cur_pos = contrastive_start_pos
        contrastive_prev_pos = 0
        do_contrastive = True 

        # cum_scores = torch.full((bsz,), 0.).cuda()
        cum_scores = [[] for beam_id in range(bsz)]

        finish_sentence = [False for beam in range(bsz)] 
        reach_answer = [False for beam in range(bsz)]
        early_stopping = [False for beam in range(bsz)]
        finish_pos = [0 for beam in range(bsz)]

        # for cur_pos in range(start_pos, total_len):
        cur_pos = start_pos
        num_batch = - (bsz // - params.max_batch_size)
        batch_past_key_values = [None for batch_id in range(num_batch)]
        batch_contrastive_past_key_values = [None for batch_id in range(num_batch)]
        while cur_pos < total_len:
            batch_next_token_log_probs1 = []
            batch_next_token_log_probs2 = []

            for batch_id in range(num_batch):
                # print('do batch:', batch_id)
                past_key_values = batch_past_key_values[batch_id]
                batch_st, batch_end = batch_id*params.max_batch_size, min((batch_id+1)*params.max_batch_size, bsz)
                logits, past_key_values = self.model.forward(tokens[batch_st:batch_end, prev_pos:cur_pos], prev_pos, past_key_values)
                batch_past_key_values[batch_id] = past_key_values

                if do_contrastive:
                    # print('do contrastive')
                    log_probs1 = F.log_softmax(logits / temperature, dim=-1)
                    log_probs1 = top_k_top_p_filtering(log_probs1, top_k=top_k, top_p=top_p)
                    next_token = torch.multinomial(torch.exp(log_probs1), num_samples=1)
                    next_token_log_probs1 = log_probs1.gather(1, next_token).reshape(-1)
                    batch_next_token_log_probs1.extend(next_token_log_probs1.tolist())

                    # contrastive_past_key_values = batch_contrastive_past_key_values[batch_id]
                    # contrastive_logits, contrastive_past_key_values = self.model.forward(contrastive_tokens[batch_st:batch_end, contrastive_prev_pos:contrastive_cur_pos], contrastive_prev_pos, contrastive_past_key_values)
                    # log_probs2 = F.log_softmax(contrastive_logits, dim=-1)
                    # next_token_log_probs2 = log_probs2.gather(1, next_token).reshape(-1)
                    # batch_next_token_log_probs2.extend(next_token_log_probs2.tolist())
                    # batch_contrastive_past_key_values[batch_id] = contrastive_past_key_values
                else:
                    next_token = torch.argmax(logits, dim=-1)

                # print('done contrastive')
                next_token = next_token.reshape(-1)
                # only replace token if prompt has already been generated
                next_token = torch.where(
                    input_text_mask[batch_st:batch_end, cur_pos], tokens[batch_st:batch_end, cur_pos], next_token
                )
                tokens[batch_st:batch_end, cur_pos] = next_token

                # if do_contrastive:
                #     contrastive_tokens[batch_st:batch_end, contrastive_cur_pos] = next_token

            for beam_id, beam in enumerate(tokens.tolist()):
                t = beam[len(prompt_tokens[beam_id]): len(prompt_tokens[beam_id]) + max_gen_len]
                # cut to eos tok if any
                try:
                    t = t[: t.index(self.tokenizer.pad_id)]
                except ValueError:
                    pass

                beam_text = self.tokenizer.decode(t)

                if do_contrastive:
                    if beam_text.endswith('.'):
                        finish_sentence[beam_id] = True
                        finish_pos[beam_id] = cur_pos
                    if not finish_sentence[beam_id]:
                        next_token_log_probs1 = batch_next_token_log_probs1[beam_id]
                        # next_token_log_probs2 = batch_next_token_log_probs2[beam_id]
                        # cum_scores[beam_id].append(next_token_log_probs1 - next_token_log_probs2)
                        cum_scores[beam_id].append(- next_token_log_probs1)

                if beam_text.endswith(answer_prefix):
                    reach_answer[beam_id] = True 

                if beam_text.endswith(stop_tokens):
                    early_stopping[beam_id] = True

            if any(reach_answer):
                do_contrastive = False

            if any(early_stopping):
                break

            if do_contrastive and all(finish_sentence):
                # print('finish sentence')
                cum_scores = [sum(beam_scores) / len(beam_scores) for beam_scores in cum_scores]
                next_sentence = torch.argmax(torch.tensor(cum_scores).cuda(), dim=-1).item()

                tokens = tokens[next_sentence].repeat((bsz, 1))
                new_batch_past_key_values = []
                for batch_id in range(num_batch):
                    mini_bsz = min((batch_id+1)*params.max_batch_size, bsz) - batch_id*params.max_batch_size
                    new_past_key_values = ()
                    for layer_past in batch_past_key_values[next_sentence // params.max_batch_size]:
                        best_keys = layer_past[0][next_sentence % params.max_batch_size].repeat(mini_bsz, 1, 1, 1)
                        best_values = layer_past[1][next_sentence % params.max_batch_size].repeat(mini_bsz, 1, 1, 1)
                        new_past_key_values += ((best_keys, best_values),)
                    new_batch_past_key_values.append(new_past_key_values)
                batch_past_key_values = new_batch_past_key_values

                # contrastive_tokens = contrastive_tokens[next_sentence].repeat((bsz, 1))
                # new_batch_past_key_values = []
                # for batch_id in range(num_batch):
                #     mini_bsz = min((batch_id+1)*params.max_batch_size, bsz) - batch_id*params.max_batch_size
                #     new_past_key_values = ()
                #     for layer_past in batch_contrastive_past_key_values[next_sentence // params.max_batch_size]:
                #         best_keys = layer_past[0][next_sentence % params.max_batch_size].repeat(mini_bsz, 1, 1, 1)
                #         best_values = layer_past[1][next_sentence % params.max_batch_size].repeat(mini_bsz, 1, 1, 1)
                #         new_past_key_values += ((best_keys, best_values),)
                #     new_batch_past_key_values.append(new_past_key_values)
                # batch_contrastive_past_key_values = new_batch_past_key_values

                cur_pos = finish_pos[next_sentence]

                cum_scores = [[] for beam_id in range(bsz)]
                finish_sentence = [False for beam in range(bsz)] 
                finish_pos = [0 for beam in range(bsz)]

            prev_pos = cur_pos
            cur_pos += 1
            if do_contrastive:
                contrastive_prev_pos = contrastive_cur_pos
                contrastive_cur_pos += 1

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[len(prompt_tokens[i]): len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.pad_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
