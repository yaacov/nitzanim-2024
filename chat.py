import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rag import retrieve_best_document  # Importing the function from rag.py

device = "cpu"
model_path = "ibm-granite/granite-3.0-8b-instruct"

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()

# Define the query
query = "How can i run a migration plan?"

# Convert augmented query into chat format
chat = [
    { "role": "user", "content": query }
]
chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# Tokenize and generate response
input_tokens = tokenizer(chat, return_tensors="pt")
output = model.generate(**input_tokens, max_new_tokens=100)
output = tokenizer.batch_decode(output, skip_special_tokens=True)

# Print the output
print("\nOutput without RAG", output[0])
