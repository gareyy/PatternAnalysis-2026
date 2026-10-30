import argparse
from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, TokenizersBackend
from peft import LoraConfig, TaskType
from tqdm import tqdm
import evaluate
import numpy as np

rouge = evaluate.load('rouge')

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
        return result.split(self.split_tok, 1)

def get_model_and_tokeniser(pull_tokeniser_from: str, pull_model_from: str, device: str, inputformat: InputFormat) -> tuple[TokenizersBackend, PreTrainedModel]:
    tokeniser = AutoTokenizer.from_pretrained(pull_tokeniser_from, device_map=device, padding_side='left')
    tokeniser.pad_token = tokeniser.eos_token
    model = AutoModelForCausalLM.from_pretrained(pull_model_from, dtype="auto")
    model.resize_token_embeddings(len(tokeniser))
    model.config.pad_token_id = tokeniser.eos_token_id
    model.loss_type = 'ForCausalLM'
    model = model.to(device)
    return tokeniser, model

# for usage in train.py
def get_pretrained_model_tokeniser_lora(args: argparse.Namespace, device: str, inputformat: InputFormat) -> tuple[TokenizersBackend, PreTrainedModel, LoraConfig | None]:
    tokeniser, model = get_model_and_tokeniser(args.model_name, args.model_name, device, inputformat)
    lora_config = None
    if args.lora:
        lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False,
                fan_in_fan_out=True,
            )
        model.add_adapter(lora_config, adapter_name="lora")
    return tokeniser, model, lora_config

def val_loss(model: PreTrainedModel, valid_data: DataLoader, device: str, max_eval_steps: int = -1):
    val_losses = []
    nbatches = 0
    for batch in tqdm(valid_data, total = len(valid_data) if max_eval_steps == -1 else max_eval_steps):
        nbatches += 1
        batch = {k: v.to(device) for k, v in batch.items()}
        # loss calcs
        with torch.amp.autocast(device, dtype=torch.bfloat16), torch.no_grad():
            outputs = model(**batch)
        val_losses.append(outputs.loss.item())
        #del outputs
        #del batch
        if max_eval_steps >= 1 and nbatches >= max_eval_steps:
            break
    return np.mean(val_losses), np.std(val_losses)

def rouge_score(model: PreTrainedModel, tokeniser: TokenizersBackend, valid_data: DataLoader, device: str, input_fmt: InputFormat, max_eval_steps: int = -1):
    nbatches = 0
    scores = {'rouge1': np.float64(0), 'rouge2': np.float64(0), 'rougeL': np.float64(0), 'rougeLsum': np.float64(0)}
    for batch in tqdm(valid_data, total = len(valid_data) if max_eval_steps == -1 else max_eval_steps):
        nbatches += 1
        str_ver = tokeniser.batch_decode(batch['input_ids'])
        str_ver = [input_fmt.extract(s.replace(tokeniser.eos_token, "")) for s in str_ver]
        reports = [s[0] for s in str_ver]
        reports = [input_fmt.fmt.format(r, "") for r in reports]
        gt_summaries = [s[1] for s in str_ver]
        report_tok = tokeniser(reports, return_tensors='pt', padding=True).to(device)
        with torch.amp.autocast(device, dtype=torch.bfloat16), torch.no_grad():
            summ_toks = model.generate(**report_tok, max_new_tokens=512)
        pred_summaries = tokeniser.batch_decode(summ_toks)
        pred_summaries = [input_fmt.extract(s.replace(tokeniser.eos_token, ""))[1] for s in pred_summaries]
        rouge_scores = rouge.compute(predictions=pred_summaries, references=gt_summaries)
        for k in scores.keys():
            scores[k] += rouge_scores[k]
        if max_eval_steps >= 1 and nbatches >= max_eval_steps:
            break
    for k in scores.keys():
        scores[k] /= nbatches
    return scores
