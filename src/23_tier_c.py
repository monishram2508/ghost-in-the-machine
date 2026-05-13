import pandas as pd
import numpy as np
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding
)

from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset, ClassLabel, Features, Value

# --- 1. SETUP & LOAD DATA ---
model_name = "distilbert-base-uncased"
print(f"🚀 Preparing {model_name} (with LoRA)...")

root = Path(__file__).parent.parent
path_human = root / "data" / "human"
path_generic = root / "data" / "generic_ai"
path_stylized = root / "data" / "stylized_ai"

# Define Paths manually to ensure they are correct strings
file = root / "data" / "fingerprint_data.csv"
df = pd.read_csv(file)

# MAP LABELS
labelmap = {"human": 0, "generic_ai": 1, "stylized_ai": 2}
df["labels"] = df["label"].map(labelmap)

# HYDRATE TEXT
def gettext(row):
    label = row["labels"]
    filename = row["filename"]
    
    # Python Path objects need to be converted to strings for open() sometimes
    if label == 0:
        path = path_human / filename
    elif label == 1:
        path = path_generic / filename
    else:
        path = path_stylized / filename
        
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return "" # Return empty string on fail

print("📖 Reading text files...")
df["text"] = df.apply(gettext, axis=1)
df = df[df["text"] != ""] # Drop empty rows
print(f"✅ Loaded {len(df)} valid text files")

# --- 2. CONVERT TO HUGGING FACE DATASET ---
# This was the missing step. We convert the whole DF at once.
# We explicitly ignore all columns except 'text' and 'labels'
hf_dataset = Dataset.from_pandas(df[['text', 'labels']])

# --- 3. TOKENIZE & CLEAN ---
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

print("⏳ Tokenizing and Sanitizing...")

# Identify columns to remove (Everything that isn't the label)
# This removes 'text', 'filename', 'ttr', etc. automatically
cols_to_remove = [col for col in hf_dataset.column_names if col != "labels"]
print(f"🗑️ Dropping dirty columns: {cols_to_remove}")

# Run Map (Tokenize + Delete dirty columns)
tokenized_datasets = hf_dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=cols_to_remove 
)

# Format for PyTorch
tokenized_datasets.set_format("torch")

# --- 4. SPLIT DATA ---
# Now we split the clean dataset into Train and Val
dataset_split = tokenized_datasets.train_test_split(test_size=0.2, seed=42)
tokenized_train = dataset_split['train']
tokenized_val = dataset_split['test']

print(f"DATASET READY. Train: {len(tokenized_train)}, Val: {len(tokenized_val)}")
print(f"Columns passing to model: {tokenized_train.column_names}")

# --- 5. MODEL SETUP (LoRA) ---
print("\n🔧 Init LoRA config")

# Num labels = 3 (Human, Generic, Styled)
base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, 
    inference_mode=False, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1,
    target_modules=["q_lin","v_lin"]
)

model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()

# --- 6. TRAINING ARGS ---
training_args = TrainingArguments(
    output_dir="./results/deberta_lora", # Renamed to match the method
    learning_rate=1e-3, # LoRA often prefers slightly higher LR (1e-3 or 5e-4)
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir='./logs',
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # FIX: changed average to 'weighted' because we have 3 classes now
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# --- 7. TRAIN ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

print("🔥 Starting Training...")
trainer.train()

# Save
model.save_pretrained("models/deberta_lora_tuned")
print("✅ Saved LoRA Model to models/deberta_lora_tuned")
