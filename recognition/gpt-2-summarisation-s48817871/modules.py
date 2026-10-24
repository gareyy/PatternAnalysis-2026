import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, TokenizersBackend
from peft import LoraConfig, TaskType

def get_model_and_tokeniser(args: argparse.Namespace, device: str) -> tuple[TokenizersBackend, PreTrainedModel, LoraConfig | None]:
    tokeniser = AutoTokenizer.from_pretrained(args.model_name, device_map=device)
    tokeniser.pad_token = tokeniser.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype="auto")
    model.resize_token_embeddings(len(tokeniser))
    model.config.pad_token_id = tokeniser.eos_token_id
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
