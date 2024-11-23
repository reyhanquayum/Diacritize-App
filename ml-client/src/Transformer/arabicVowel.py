import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# Step 1: Convert the XLS file to CSV format
df = pd.read_excel("ArabLEX.xls")
df.to_csv("ArabLEX.csv", index=False)

# Step 2: Load the CSV data and preprocess it
with open("ArabLEX.csv", "r", encoding="utf-8") as file:
    arab_lex_data = [line.strip().split(",") for line in file if line[0][0]=="V"]

# Split the dataset into training and evaluation sets
train_data, eval_data = train_test_split(arab_lex_data, test_size=0.1, random_state=42)

# Step 3: Train a model to predict vowels
model_name = "asafaya/bert-base-arabic"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def train_model(train_data, eval_data, n_iter=10):
    max_seq_length = 128  
    vowel_to_id = {
        "": 0,
        "ف": 1,
        "ر": 2,
        "ي": 3
    }
    examples_train = []
    examples_eval = []
    
    for data in [train_data, eval_data]:
        examples = []
        for row in data:
            text = row[4]
            vowel_label = row[5]
            text = text.strip()  
            inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_seq_length)
            label_id = vowel_to_id.get(vowel_label, 0) 
            example = {
                "input_ids": inputs["input_ids"][0],
                "attention_mask": inputs["attention_mask"][0],
                "labels": torch.tensor(label_id)
            }
            examples.append(example)
        if data is train_data:
            examples_train = examples
        else:
            examples_eval = examples

    training_args = TrainingArguments(
        output_dir="./output_dir",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=n_iter,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=examples_train,
        eval_dataset=examples_eval  # Provide the evaluation dataset
    )
    
    trainer.train()
    return trainer.model

# Train the model
trained_model = train_model(train_data, eval_data)
