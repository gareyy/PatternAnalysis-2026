# TODO: like fix this up
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import modules
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(add_help=True)

parser.add_argument("--input-folder", default="./modelout", help='Where to load pretrained model from')
parser.add_argument("--model-name", default="openai-community/gpt2", help='Name of base model, used for getting tokeniser')

args = parser.parse_args()
inputfmt = modules.TLDRFormat()
tokeniser, model, = modules.get_model_and_tokeniser(args.model_name, args.input_folder, device, inputfmt)
prompt = "Right parahilar infiltrate and atelectasis. Increased retrocardiac density related to atelectasis and consolidation associated with right pleural effusion. Clinical data is important for correct radiological assessment. TLDR; "
with torch.no_grad():
    inputs = tokeniser(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512)
print(tokeniser.decode(outputs[0], skip_special_tokens=True))
