from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
from dataset import BioLaySummDataset
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 6
NUM_EPOCHS = 1
MAX_STEPS = 100

model_name = "openai-community/gpt2-medium"
model_name = "openai-community/gpt2"
tokeniser = AutoTokenizer.from_pretrained(model_name)
tokeniser.pad_token = tokeniser.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
model.resize_token_embeddings(len(tokeniser))
model.config.pad_token_id = tokeniser.eos_token_id
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
            loss = outputs.loss
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        pbar.update()
        steps += 1
        if steps >= MAX_STEPS:
            break
    if steps >= MAX_STEPS:
        break
model.save_pretrained("./modelout", from_pt=True)
