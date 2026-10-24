# TODO: like fix this up
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(add_help=True)

parser.add_argument("--input-folder", default="./modelout", help='Where to load pretrained model from')

args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(
    #"openai-community/gpt2-medium",
    #"openai-community/gpt2",
    args.input_folder,
    device_map=device,
)

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
prompt = "Right parahilar infiltrate and atelectasis. Increased retrocardiac density related to atelectasis and consolidation associated with right pleural effusion. Clinical data is important for correct radiological assessment. TLDR;"
with torch.no_grad():
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
