# TODO: clean this up
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, default_data_collator
from dataset import BioLaySummDataset

model_name = "openai-community/gpt2-medium"
model_name = "openai-community/gpt2"
tokeniser = AutoTokenizer.from_pretrained(model_name)
tokeniser.pad_token = tokeniser.eos_token
train_dataset = BioLaySummDataset("train", tokeniser)
valid_dataset = BioLaySummDataset("validation", tokeniser)

model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
model.resize_token_embeddings(len(tokeniser))
model.config.pad_token_id = tokeniser.eos_token_id

training_args = TrainingArguments(
    output_dir="gpt2-supervised-summarize-checkpoint",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=1,
    eval_strategy="epoch",
    learning_rate=1e-5,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=default_data_collator,
)

trainer.train()

