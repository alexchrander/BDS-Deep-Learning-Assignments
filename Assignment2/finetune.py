# Load dependencies
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import classification_report
from pathlib import Path

# Fixed settings as recommended by the assignment
SEED       = 42
MODEL_NAME = "AI-Growth-Lab/PatentSBERTa"
MAX_LEN    = 256   # max token length per claim
BATCH_SIZE = 16    # number of claims processed at once
EPOCHS     = 1     # assignment specifies fine-tune once
LR         = 2e-5  # learning rate within recommended range

torch.manual_seed(SEED)

class PatentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        # Tokenize all texts at once — truncate to MAX_LEN and pad shorter ones
        self.encodings = tokenizer(
            texts,
            truncation=True,  # cut off anything longer than MAX_LEN
            padding=True,     # pad shorter texts to the same length in the batch
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        # Convert labels to a tensor so PyTorch can use them during training
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        # Returns total number of examples — needed by DataLoader
        return len(self.labels)

    def __getitem__(self, idx):
        # Returns a single example as a dict — DataLoader calls this per batch
        return {
            "input_ids":      self.encodings["input_ids"][idx],       # tokenized text
            "attention_mask": self.encodings["attention_mask"][idx],  # 1 for real tokens, 0 for padding
            "labels":         self.labels[idx]                        # gold label for this example
        }
    
# Load gold training set (train_silver + gold_100) and eval set
df_train = pd.read_parquet("parquet/train_gold.parquet")
df_eval  = pd.read_parquet("parquet/eval_silver.parquet")

# Load tokenizer from PatentSBERTa
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Create dataset objects for train and eval
train_dataset = PatentDataset(
    texts=df_train["text"].tolist(),
    labels=df_train["is_green_gold"].astype(int).tolist(),
    tokenizer=tokenizer
)
eval_dataset = PatentDataset(
    texts=df_eval["text"].tolist(),
    labels=df_eval["is_green_silver"].astype(int).tolist(),
    tokenizer=tokenizer
)

# DataLoader batches the data and shuffles training set each epoch
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_loader  = DataLoader(eval_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# Load model with a binary classification head (num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

# AdamW is the standard optimizer for fine-tuning transformers
optimizer = AdamW(model.parameters(), lr=LR)

# Linear warmup scheduler — gradually increases lr at start to stabilize training
total_steps   = len(train_loader) * EPOCHS
warmup_steps  = int(0.1 * total_steps)  # 10% of steps used for warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print("Starting fine-tuning...")

model.train()
for epoch in range(EPOCHS):
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        # Move batch to GPU
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        # Forward pass — compute predictions and loss
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss

        # Backward pass — compute gradients and update weights
        optimizer.zero_grad()  # clear gradients from previous batch
        loss.backward()        # compute gradients
        optimizer.step()       # update weights
        scheduler.step()       # update learning rate

        total_loss += loss.item()

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"Batch {batch_idx + 1}/{len(train_loader)} — Loss: {loss.item():.4f}")

    print(f"Epoch {epoch + 1} complete — Avg loss: {total_loss / len(train_loader):.4f}")

def evaluate(loader, description):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():  # no gradient computation needed during eval
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=1)  # pick highest scoring class

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"\n--- {description} ---")
    print(classification_report(all_labels, all_preds, target_names=["not green", "green"]))

# Evaluate on eval_silver (comparable to Part A baseline)
evaluate(eval_loader, "eval_silver")

# Evaluate on gold_100
df_gold      = pd.read_csv("csv/hitl_human_labeled.csv")
gold_dataset = PatentDataset(
    texts=df_gold["text"].tolist(),
    labels=df_gold["is_green_human"].astype(int).tolist(),
    tokenizer=tokenizer
)
gold_loader = DataLoader(gold_dataset, batch_size=BATCH_SIZE, shuffle=False)
evaluate(gold_loader, "gold_100")

# Save fine-tuned model and tokenizer
OUTPUT_DIR = Path("models/patentsberta-finetuned")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nModel saved to {OUTPUT_DIR}")