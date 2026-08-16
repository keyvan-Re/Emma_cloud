from transformers.generation.logits_process import LogitsProcessor
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    set_seed,
)
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
import torch, os

max_chunk_overlap = 20
pre_seq_len = 128
prefix_projection = False

class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores