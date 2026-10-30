# TODO: like fix this up
import torch
from torch.utils.data import DataLoader
from dataset import BioLaySummDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, DataCollatorWithPadding
import argparse
import modules
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from time import perf_counter

parser = argparse.ArgumentParser(add_help=True)

parser.add_argument("--input-folder", default="./modelout", help='Where to load pretrained model from')
parser.add_argument("--model-name", default="openai-community/gpt2", help='Name of base model, used for getting tokeniser')

args = parser.parse_args()
inputfmt = modules.TLDRFormat()
tokeniser, model, = modules.get_model_and_tokeniser(args.model_name, args.input_folder, device, inputfmt)
data_collator = DataCollatorWithPadding(tokeniser)
torch.no_grad()
model.eval()
valid_dataset = BioLaySummDataset("validation", tokeniser, inputfmt.fmt)
valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=16, collate_fn=data_collator)
model = model.to(device)
modules.val_loss(model, valid_dataloader, device, max_eval_steps=10)
modules.rouge_score(model, tokeniser, valid_dataloader, device, inputfmt, max_eval_steps=10)
prompt = "Right parahilar infiltrate and atelectasis. Increased retrocardiac density related to atelectasis and consolidation associated with right pleural effusion. Clinical data is important for correct radiological assessment."
prompt = inputfmt.fmt.format(prompt, "")
inputs = tokeniser(prompt, return_tensors="pt").to(model.device)
streamer = TextStreamer(tokeniser)
start = perf_counter()
output_tokens = model.generate(**inputs, streamer=streamer, max_new_tokens=512)
end = perf_counter()
num_tokens_generated = output_tokens.shape[1]
print(f"{num_tokens_generated / (end-start):.2f} tok/s")
out = tokeniser.decode(output_tokens[0], skip_special_tokens=True)
prompt, predict = inputfmt.extract(out)
print(predict)
