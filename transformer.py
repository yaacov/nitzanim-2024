# pylint: disable=all

import logging
import torch
from transformers import BertTokenizer, BertModel, pipeline

# Suppress logging information from transformers and other libraries
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Load a pipeline for question answering
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Function to get embeddings from BERT for each word in the answer
def get_bert_embeddings(words):
    inputs = tokenizer(words, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Context and question
context = """
The following prerequisites apply to all migrations:
IP addresses, VLANs, and other network configuration settings must not be changed before or during migration.
The MAC addresses of the virtual machines are preserved during migration.
The network connections between the source environment, the OpenShift Virtualization cluster,
and the replication repository must be reliable and uninterrupted.
If you are mapping more than one source and destination network,
you must create a network attachment definition for each additional destination network.
"""
question = "What are some prerequisites of a migration?"

# Generate an answer using the QA pipeline
result = qa_pipeline(question=question, context=context)
answer = result['answer']

# Print the context, question, and answer
print(f"Context: {context}")
print(f"Question: {question}")
print(f"Answer: {answer}\n")

# Get embeddings for each word in the answer and print them
words = answer.split()
for word in words:
    embeddings = get_bert_embeddings(word).squeeze().detach().numpy()
    first_part = embeddings[:5]
    last_part = embeddings[-5:]
    print(f"Word: \"{word}\"\nEmbeddings shape: {embeddings.shape}\n"
          f"{first_part.tolist()} ... {last_part.tolist()}\n")
