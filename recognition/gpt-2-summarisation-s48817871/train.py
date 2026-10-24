from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
from transformers.loss.loss_utils import ForCausalLMLoss
from dataset import BioLaySummDataset
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from peft import LoraConfig, TaskType
from shutil import rmtree
import os
device = "cuda" if torch.cuda.is_available() else "cpu"

EMUDHRA_PRIVATE_KEY = 4
torch.manual_seed(EMUDHRA_PRIVATE_KEY)

BATCH_SIZE = 6
NUM_EPOCHS = 1
MAX_STEPS = 100
SAVE_EVERY = 100
OUTPUT_FOLDER = "./modelout"
LORA_TRAIN = True

# TODO: add cmdline parameter stuff here

if os.path.exists(OUTPUT_FOLDER):
    print(f"Previous folder {OUTPUT_FOLDER} exists, removing...")
    rmtree(OUTPUT_FOLDER)

model_name = "openai-community/gpt2"
tokeniser = AutoTokenizer.from_pretrained(model_name)
tokeniser.pad_token = tokeniser.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")

if LORA_TRAIN:
    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False)
    model.add_adapter(lora_config, adapter_name="lora")

model.resize_token_embeddings(len(tokeniser))
model.config.pad_token_id = tokeniser.eos_token_id
model.loss_type = 'ForCausalLM'
data_collator = DataCollatorWithPadding(tokeniser)

train_dataset = BioLaySummDataset("train", tokeniser)
valid_dataset = BioLaySummDataset("validation", tokeniser)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator)
valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=BATCH_SIZE, collate_fn=data_collator)

n_training_steps = len(train_dataloader) * NUM_EPOCHS
optimiser = optim.AdamW(model.parameters(), lr=5e-5)
scheduler = lr_scheduler.CosineAnnealingLR(optimiser, T_max=NUM_EPOCHS)

pbar = tqdm(range(n_training_steps))
model.to(device)
steps = 0
model.train()
for e in range(NUM_EPOCHS):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast(device, dtype=torch.bfloat16):
            outputs = model(**batch)
            # model uses its own loss function that is bundled with outputs, but i've decided to go here with ForCausalLMLoss
            loss = outputs.loss
            # equivalent to `loss = ForCausalLMLoss(outputs.logits, batch['labels'], tokeniser.vocab_size)`
            # under the hood, it uses cross entropy loss
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        pbar.update()
        steps += 1
        if MAX_STEPS != -1 and steps >= MAX_STEPS:
            break
        if steps % SAVE_EVERY == 0:
            print(f"Saving {'lora+model' if LORA_TRAIN else 'model'} at {OUTPUT_FOLDER}")
            model.save_pretrained(OUTPUT_FOLDER)
    if MAX_STEPS != -1 and steps >= MAX_STEPS:
        break
    scheduler.step()
print("TRAINING FINISHED")
print(f"Saving {'lora+model' if LORA_TRAIN else 'model'} at {OUTPUT_FOLDER}")
model.save_pretrained(OUTPUT_FOLDER)
