# pylint: disable=all

import logging
import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine

# Suppress logging information from transformers and other libraries
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# Load pre-trained BERT model and tokenizer from transformers
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get embeddings from BERT
def get_bert_embeddings(word):
    inputs = tokenizer(word, return_tensors='pt')
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# List of words and the word to match
words = ["car", "orange", "tree", "grape", "fruit", "cat", "giraffe", "avocado"]
word_to_match = "apple"

# Get embeddings for the word to match
match_embeddings = get_bert_embeddings(word_to_match).squeeze().detach().numpy()

# Store the best match details
best_match_word = None
best_match_score = float('inf')

# Calculate cosine similarity and print results
for word in words:
    word_embeddings = get_bert_embeddings(word).squeeze().detach().numpy()
    similarity = cosine(word_embeddings, match_embeddings)
    
    # Extract the first 5 and last 5 numbers from the embeddings
    first_part = word_embeddings[:5]
    last_part = word_embeddings[-5:]
    
    # Print each word with its embeddings' shape, first and last 5 numbers
    print(f"Word: \"{word}\"\nEmbeddings shape: {word_embeddings.shape}\n"
          f"{first_part.tolist()} ... {last_part.tolist()},\nSimilarity: {1 - similarity}\n")
    
    # Update best match if the current similarity is better
    if similarity < best_match_score:
        best_match_score = similarity
        best_match_word = word

# Print the word with the best match
print(f"\nBest match for {word_to_match}: {best_match_word} with similarity: {1 - best_match_score}")
