from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
from dataset import BioLaySummDataset
import modules
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from peft import LoraConfig, TaskType
import shutil
import os
import argparse
device = "cuda" if torch.cuda.is_available() else "cpu"

EMUDHRA_PRIVATE_KEY = 4
torch.manual_seed(EMUDHRA_PRIVATE_KEY)

parser = argparse.ArgumentParser(add_help=True)

parser.add_argument("--output-folder", default="./modelout", type=str, help='Where to store output of training')
parser.add_argument("--batch-size", default=6, type=int, help="Batch size in training")
parser.add_argument("--save-every", default=100, type=int, help="Save every n steps, also does evaluation of metrics (val loss, train loss, ROUGE)")
# TODO: change to -1 when not using for testing
parser.add_argument("--max-steps", default=100, type=int, help="Max number of steps to train for (independent of epoch amount, takes priority over epoch amount), -1 to train for n epochs * number of batches in dataset")
parser.add_argument("--n-epochs", default=1, type=int, help="Number of epochs to train for")
parser.add_argument("--model-name", default="openai-community/gpt2", type=str, help="Huggingface model to train from")

parser.add_argument("--lora", action='store_true', help='Train a LoRA instead of a finetuned model')

args = parser.parse_args()

if os.path.exists(args.output_folder):
    print(f"Previous folder {args.output_folder} exists, removing...")
    shutil.rmtree(args.output_folder)

if args.lora:
    print("Training with LoRA...")
else:
    print("Finetuning Model Weights...")

tokeniser, model, _ = modules.get_model_and_tokeniser(args, device)

data_collator = DataCollatorWithPadding(tokeniser)

train_dataset = BioLaySummDataset("train", tokeniser)
valid_dataset = BioLaySummDataset("validation", tokeniser)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=6, collate_fn=data_collator)
valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=6, collate_fn=data_collator)

n_training_steps = len(train_dataloader) * args.n_epochs if args.max_steps == -1 else args.max_steps
optimiser = optim.AdamW(model.parameters(), lr=5e-5)
scheduler = lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.n_epochs)

pbar = tqdm(range(n_training_steps))
model.to(device)
steps = 0
model.train()
for e in range(args.n_epochs):
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
        if args.max_steps != -1 and steps >= args.max_steps:
            break
        if steps % args.save_every == 0:
            print(f"Saving {'lora+model' if args.lora else 'model'} at {args.output_folder}")
            model.save_pretrained(args.output_folder)
    if args.max_steps != -1 and steps >= args.max_steps:
        break
    scheduler.step()
print("TRAINING FINISHED")
print(f"Saving {'lora+model' if args.lora else 'model'} at {args.output_folder}")
model.save_pretrained(args.output_folder)
