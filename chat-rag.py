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

# Define documents for retrieval
documents = [
    "The Migration Toolkit for Virtualization (MTV) is provided as an OpenShift Container Platform Operator. It creates and manages the following custom resources (CRs) and services.",
    "You run a migration plan by creating a Migration CR that references the migration plan. If a migration is incomplete, you can run a migration plan multiple times until all VMs are migrated.",
    "For each VM in the migration plan, the Migration Controller creates a VirtualMachineImport CR and monitors its status. When all VMs have been migrated, the Migration Controller sets the status of the migration plan to Completed."
]

# Use retrieve_best_document to find the best-matching document
best_document = retrieve_best_document(documents, query)

# Augment the query with the best-matching document
augmented_query = f"{best_document} {query}"

# Convert augmented query into chat format
chat = [
    { "role": "user", "content": augmented_query }
]
chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# Tokenize and generate response
input_tokens = tokenizer(chat, return_tensors="pt")
output = model.generate(**input_tokens, max_new_tokens=100)
output = tokenizer.batch_decode(output, skip_special_tokens=True)

# Print the output
print("\nOutput (RAG):", output[0])
