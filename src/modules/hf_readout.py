"""Local Hugging Face causal LMs as decision models (LLM-as-Jev readout).

Used for open Jev-style models and their Qwen backbones. Each option is scored
by its conditional log-likelihood after the prompt; probabilities are a softmax
over option scores. ``length_norm="mean"`` divides by option token count to
reduce the bias toward short labels (relevant for multi-token species names).

The prompt shares one KV cache across options: the prompt is encoded once and
options are scored in batches against the cached prefix.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from src.modules.base import DecisionModel, DecisionRequest, ModelOutput
from src.modules.prompts import render_text_prompt


class HFOptionScorer(DecisionModel):
    supports_images = False

    def __init__(
        self,
        model,
        tokenizer,
        *,
        name: str,
        length_norm: Literal["sum", "mean"] = "mean",
        use_chat_template: bool = True,
        answer_prefix: str = "Answer:",
        option_prefix: str = " ",
        option_batch_size: int = 64,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.name = name
        self.length_norm = length_norm
        self.use_chat_template = use_chat_template and tokenizer.chat_template is not None
        self.answer_prefix = answer_prefix
        # BPE tokenizers attach the leading space to the next word, so the prompt
        # ends without one and each option is scored as option_prefix + option.
        self.option_prefix = option_prefix
        self.option_batch_size = option_batch_size

    @classmethod
    def from_pretrained(cls, model_id: str, *, dtype=torch.bfloat16, device_map="auto", **kwargs):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device_map)
        return cls(model, tok, name=kwargs.pop("name", model_id), **kwargs)

    def _prompt_ids(self, request: DecisionRequest) -> list[int]:
        text = render_text_prompt(request)
        if self.use_chat_template:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
            )
            return self.tokenizer(text + self.answer_prefix, add_special_tokens=False)["input_ids"]
        return self.tokenizer(text + "\n" + self.answer_prefix)["input_ids"]

    @torch.no_grad()
    def option_logprobs(self, request: DecisionRequest) -> np.ndarray:
        """Summed (or mean) log-probability of each option's tokens given the prompt."""
        device = self.model.device
        prompt = torch.tensor([self._prompt_ids(request)], device=device)
        out = self.model(prompt, use_cache=True)
        past = out.past_key_values
        first_logprobs = torch.log_softmax(out.logits[0, -1].float(), dim=-1)

        opt_ids = [
            self.tokenizer(self.option_prefix + o, add_special_tokens=False)["input_ids"]
            for o in request.options
        ]
        scores = np.empty(len(opt_ids))
        for start in range(0, len(opt_ids), self.option_batch_size):
            chunk = opt_ids[start:start + self.option_batch_size]
            width = max(len(ids) for ids in chunk)
            pad = self.tokenizer.pad_token_id or 0
            batch = torch.tensor([ids + [pad] * (width - len(ids)) for ids in chunk], device=device)
            mask = torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids in chunk], device=device)

            # First option token is predicted by the prompt's last position.
            lp = first_logprobs[batch[:, 0]].clone()
            if width > 1:
                b = len(chunk)
                batch_past = _expand_cache(past, b)
                attn = torch.cat([torch.ones(b, prompt.shape[1], device=device, dtype=mask.dtype), mask], dim=1)
                logits = self.model(batch, past_key_values=batch_past, attention_mask=attn).logits
                tok_lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
                tok_lp = tok_lp.gather(-1, batch[:, 1:, None]).squeeze(-1) * mask[:, 1:]
                lp = lp + tok_lp.sum(dim=1)
            if self.length_norm == "mean":
                lp = lp / mask.sum(dim=1)
            scores[start:start + len(chunk)] = lp.cpu().numpy()
        return scores

    def predict(self, request: DecisionRequest) -> ModelOutput:
        self.check(request)
        scores = self.option_logprobs(request)
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        return ModelOutput(probs=probs, raw={"option_scores": scores.tolist(), "length_norm": self.length_norm})


def _expand_cache(past, batch_size: int):
    """Repeat a batch-1 KV cache to ``batch_size`` (copy, so the prefix cache stays reusable)."""
    from transformers import DynamicCache

    if isinstance(past, DynamicCache):
        legacy = past.to_legacy_cache()
        return DynamicCache.from_legacy_cache(
            tuple((k.expand(batch_size, *k.shape[1:]).contiguous(), v.expand(batch_size, *v.shape[1:]).contiguous()) for k, v in legacy)
        )
    return tuple(
        (k.expand(batch_size, *k.shape[1:]).contiguous(), v.expand(batch_size, *v.shape[1:]).contiguous())
        for k, v in past
    )
