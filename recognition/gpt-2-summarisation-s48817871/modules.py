import argparse
from typing import overload
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, TokenizersBackend
from peft import LoraConfig, TaskType

class InputFormat:
    fmt = "{} {}"
    split_tok = ""
    special_tokens = []

    def extract(self, result: str):
        return []

class TLDRFormat(InputFormat):
    fmt = "{} TLDR; {}"
    split_tok = " TLDR; "
    special_tokens = []

    def extract(self, result: str):
        return result.split(self.split_tok)

def get_model_and_tokeniser(pull_tokeniser_from: str, pull_model_from: str, device: str, inputformat: InputFormat) -> tuple[TokenizersBackend, PreTrainedModel]:
    tokeniser = AutoTokenizer.from_pretrained(pull_tokeniser_from, device_map=device)
    tokeniser.pad_token = tokeniser.eos_token
    model = AutoModelForCausalLM.from_pretrained(pull_model_from, dtype="auto")
    model.resize_token_embeddings(len(tokeniser))
    model.config.pad_token_id = tokeniser.eos_token_id
    return tokeniser, model

# for usage in train.py
def get_pretrained_model_tokeniser_lora(args: argparse.Namespace, device: str, inputformat: InputFormat) -> tuple[TokenizersBackend, PreTrainedModel, LoraConfig | None]:
    tokeniser, model = get_model_and_tokeniser(args.model_name, args.model_name, device, inputformat)
    model.loss_type = 'ForCausalLM'
    lora_config = None
    if args.lora:
        lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False,
                fan_in_fan_out=True,
            )
        model.add_adapter(lora_config, adapter_name="lora")
    return tokeniser, model, lora_config
