import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-large-arabertv02")
model = AutoModelForTokenClassification.from_pretrained("aubmindlab/bert-large-arabertv02")

# Mapping of consonants to vowels
consonant_vowel_mapping = {
    "ف": "i",
    "ر": "a",
    "ي": "u"
}

# Read input text from file
input_file = "quran-simple-no-ayat-numbers-with-vowels-3.txt"
with open(input_file, "r", encoding="utf-8") as file:
    text = file.read()

# Split the text into sentences using asterisk (*) as separator
sentences = text.split("*")

# Initialize list to store sentences with vowels
sentences_with_vowels = []

# Tokenize each sentence, predict vowels, and add them to the list
for sentence in sentences:
    # Tokenize the sentence
    tokens = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
    
    # Predict vowels
    with torch.no_grad():
        outputs = model(**tokens)
    
    # Decode the tokens and retrieve the vowels
    decoded_tokens = tokenizer.decode(tokens.input_ids[0], skip_special_tokens=True)
    vowels = []
    for token, prediction in zip(decoded_tokens.split(), outputs.logits.argmax(2)[0]):
        consonant = token[0]  # Get the consonant
        vowel = consonant_vowel_mapping.get(consonant, "")  # Get the corresponding vowel from the mapping
        vowels.append(token + vowel)  # Add the vowel to the token
    
    # Join tokens with predicted vowels back to form the sentence
    sentence_with_vowels = " ".join(vowels)
    sentences_with_vowels.append(sentence_with_vowels)

# Write the sentences with predicted vowels to a new file
output_file = "output_with_vowels.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(sentences_with_vowels))
