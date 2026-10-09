# TODO: like fix this up
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = AutoModelForCausalLM.from_pretrained(
    "gpt2-finetunedtest/checkpoint-155",
    device_map='auto',
)

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
with torch.no_grad():
    prompt = ""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
