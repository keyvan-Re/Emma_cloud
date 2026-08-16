from transformers import Trainer, HfArgumentParser
from dataclasses import dataclass, field
@dataclass
class DataArguments:
    memory_search_top_k: int = field(default=2)
    memory_basic_dir: str = field(default='MMPL_gpt/memories')
    memory_file: str = field(default='update_memory_0512_eng.json')
    language: str = field(default='en')
    max_history: int = field(default=7,metadata={"help": "maximum number for keeping current history"},)
    enable_forget_mechanism: bool = field(default=False)

@dataclass
class ModelArguments:
    """
   model_type: str = field(
        default="chatglm",
        metadata={"help": "model type: chatglm / belle"},
    )
    base_model: str = field(
        default="THUDM/chatglm-6b",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    """
    
    adapter_model: str = field(
        default="MMPL_gpt",
        metadata={"help": ""},
    )
    ptuning_checkpoint: str = field(
        default=None,
        metadata={"help": "Path to pretrained prefix embedding of ptuning"},
    )
    
    

    

data_args,model_args = HfArgumentParser(
    (DataArguments,ModelArguments)
).parse_args_into_dataclasses()
"""
data_args, model_args, _ = HfArgumentParser(
    (DataArguments, ModelArguments)
).parse_args_into_dataclasses(return_remaining_strings=True)
"""