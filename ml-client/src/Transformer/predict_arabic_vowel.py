import os
import spacy
import random
from spacy.training import Example
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load the Arabert tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-large-arabertv02")
model = AutoModelForSequenceClassification.from_pretrained("aubmindlab/bert-large-arabertv02")

# Define the training data
TRAIN_DATA = [
    ("ف", "i"),  # Example: the vowel "i" follows the consonant "ف"
    ("ر", "a"),  # Example: the vowel "a" follows the consonant "ر"
    ("ي", "u"),  # Example: the vowel "u" follows the consonant "ي"
    # Add more training examples as needed
]

# Define the training loop
def train_model(train_data, n_iter=10):
    random.shuffle(train_data)
    examples = []
    for text, annotation in train_data:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        label = {"label": annotation}
        examples.append(Example.from_dict(inputs, label))
    
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

# Train the model
trained_model = train_model(TRAIN_DATA)

# Test the model
def predict_vowel(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    logits = trained_model(**inputs).logits
    predicted_vowel_index = logits.argmax().item()
    predicted_vowel = tokenizer.decode(predicted_vowel_index)
    return predicted_vowel

# Function to predict vowels for texts in files in a given directory
def predict_vowels_in_directory(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as file:
                text = file.read()
                predicted_vowel = predict_vowel(text)
                print(f"Predicted vowel for '{filename}': {predicted_vowel}")

# Specify the directory containing the files to test
test_directory = "your_test_directory_path_here"
predict_vowels_in_directory(test_directory)

