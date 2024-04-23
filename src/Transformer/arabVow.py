import os
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from spacy.training import Example  # Import the Example class

# Load the Arabert tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-large-arabertv02")
model = AutoModelForSequenceClassification.from_pretrained("aubmindlab/bert-large-arabertv02")

# Define the training loop
def train_model(train_data, n_iter=10):
    examples = []
    for idx, row in train_data.iterrows():
        sent1 = row["sent1"]
        sent2 = row["sent2"]
        label = row["label"]
        
        inputs = tokenizer(sent1, sent2, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].squeeze()  
        attention_mask = inputs["attention_mask"].squeeze()  
        token_type_ids = inputs["token_type_ids"].squeeze()  
        
        # Corrected the use of Example.from_dict method
        example = Example.from_dict(
            {
                "input_ids": input_ids.tolist(),  # Convert tensor to list
                "attention_mask": attention_mask.tolist(),  # Convert tensor to list
                "token_type_ids": token_type_ids.tolist(),  # Convert tensor to list
            }, 
            {
                "label": torch.tensor([label])  # Keep label as tensor
            }
        )
        examples.append(example)
    
    training_args = TrainingArguments(
        per_device_train_batch_size=8,
        num_train_epochs=n_iter,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=examples,
    )
    
    trainer.train()
    return trainer.model


# Load the dev data from dev.txt
dev_data = pd.read_csv("dev.txt", sep="\t")

# Train the model
trained_model = train_model(dev_data)

# Function to predict labels for texts in files in a given directory
def predict_labels_in_directory(directory, output_file):
    with open(output_file, "w", encoding="utf-8") as outfile:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                with open(filepath, "r", encoding="utf-8") as file:
                    lines = file.readlines()
                    for line in lines:
                        parts = line.strip().split("\t")
                        if len(parts) == 3:
                            id_, sent1, sent2 = parts
                            inputs = tokenizer(sent1, sent2, return_tensors="pt", padding=True, truncation=True)
                            logits = trained_model(**inputs).logits
                            predicted_label_index = logits.argmax().item()
                            outfile.write(f"{id_}\t{predicted_label_index}\n")

# Specify the directory containing the files to test
test_directory = "Test.txt"
output_file = "test-diactric.txt"
predict_labels_in_directory(test_directory, output_file)

# Define a function to add the values
def add_values(a, b):
    return a + b

# Sample line from the Quran text
quran_line = "خَتَمَ اللَّهُ عَلَىٰ قُلُوبِهِمْ وَعَلَىٰ سَمْعِهِمْ وَعَلَىٰ أَبْصَارِهِمْ غِشَاوَةٌ وَلَهُمْ عَذَابٌ عَظِيمٌ"

# Tokenize the text
tokens = tokenizer(quran_line, padding=True, truncation=True, return_tensors="pt", max_length=512)

# Predict vowels
with torch.no_grad():
    outputs = model(**tokens)

vowels = []
for token, prediction in zip(tokens["input_ids"][0], outputs.logits.argmax(2)[0]):
    token = tokenizer.decode(token.item())
    vowels.append(token)
    vowels[-1] += " " if prediction == 1 else ""
    
text_with_vowels = "\n".join(vowels)
with open("quran_simple_no_ayat_numbers_with_vowels-4.txt", "w", encoding="utf-8") as f:
    f.write(text_with_vowels)
