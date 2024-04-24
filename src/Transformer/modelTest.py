import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the trained model from the checkpoint files
model_name = "asafaya/bert-base-arabic"
# Define the directory containing the output folders
output_dir = "./output_dir"

# Function to predict labels for text data
def predict_labels(text_data, model, tokenizer):
    max_seq_length = 128
    labels = []

    # Tokenize and preprocess the text data
    inputs = tokenizer(text_data, return_tensors="pt", padding="max_length", truncation=True, max_length=max_seq_length)

    # Use the trained model to predict labels
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Post-process the predicted labels
    predicted_labels = torch.argmax(logits, dim=1).tolist()
    print("&*(@))#")
    for label_id in predicted_labels:
        # Map label IDs to actual labels
        # This mapping should match the one used during training
        if label_id == 0:
            label = ""  # No vowel
        elif label_id == 1:
            label = "ف"
        elif label_id == 2:
            label = "ر"
        elif label_id == 3:
            label = "ي"
        else:
            label = ""  # Handle unknown labels
        labels.append(label)
        print("*")

    return labels

# Function to load and preprocess text data from a file
def load_text_data(file_path):
    print(file_path)
    with open(file_path, "r", encoding="utf-8") as file:
        print("@#$")
        text_data = [line.strip() for line in file]
    return text_data

# Function to save labeled text data to a file
def save_labeled_text_data(text_data, predicted_labels, output_file):
    with open(output_file, "w", encoding="utf-8") as file:
        for text, label in zip(text_data, predicted_labels):
            file.write(f"{text} {label}\n")

# Initialize variables to track the best evaluation loss and corresponding folder
best_eval_loss = float('inf')
best_model_path = None

# Iterate through all subdirectories in the output directory
for folder_name in os.listdir(output_dir):
    print("Folder name:", folder_name)
    folder_path = os.path.join(output_dir, folder_name)
    # Check if the item in the directory is a directory itself and contains a config.json file
    if os.path.isdir(folder_path) and "config.json" in os.listdir(folder_path):
        # Load and preprocess evaluation results from the trainer_state.json file
        trainer_state_path = os.path.join(folder_path, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, "r") as state_file:
                trainer_state = json.load(state_file)
                # Extract the evaluation loss from the log history
                log_history = trainer_state.get("log_history", [])
                for log_entry in log_history:
                    eval_loss = log_entry.get("eval_loss")
                    if eval_loss is not None and eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        best_model_path = folder_path

# Print the path of the best performing model
if best_model_path:
    print("Best performing model path:", best_model_path)
else:
    print("No folder with the required config.json file found.")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the best model
if best_model_path:
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
    
    # Load and preprocess text data from each file
    file_paths = ["Train.txt", "Test.txt", "Dev.txt"]
    for file_path in file_paths:
        print("Processing file:", file_path)
        # Load text data from the file
        text_data = load_text_data(file_path)
        print("*")
        # Predict labels for the text data
        predicted_labels = predict_labels(text_data, model, tokenizer)
        print("Back from labels")
        # Save labeled text data to a file
        output_file = f"{os.path.splitext(file_path)[0]}_labeled.txt"
        save_labeled_text_data(text_data, predicted_labels, output_file)

        print(f"Labeled text data saved to {output_file}")
else:
    print("No folder with the required config.json file found.")
