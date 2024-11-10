from sentence_transformers import SentenceTransformer, util

# Load a model specifically optimized for semantic similarity
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def retrieve_best_document(context, query):
    """
    Retrieve the document from context that best matches the query.
    
    Parameters:
    - context (list of str): List of documents to search within.
    - query (str): The query to match against the documents.
    
    Returns:
    - str: The document with the highest similarity to the query.
    """
    # Encode the context documents into embeddings
    doc_embeddings = model.encode(context, convert_to_tensor=True)
    
    # Get embedding for the query
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Calculate cosine similarity with each document embedding
    similarities = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
    
    # Find the index of the most similar document
    best_match_index = similarities.argmax().item()
    
    # Retrieve and return the best matching document
    best_document = context[best_match_index]
    return best_document

if __name__ == "__main__":
    # Example usage
    documents = [
        "Dogs are loyal, intelligent animals known for their companionship and protective nature. They come in a wide range of breeds, each with unique traits and personalities, and are often trained to assist humans in various roles, from pets to service and working animals.",
        "Cats are independent and curious animals, appreciated for their playful yet relaxed demeanor. Known for their agility and keen hunting instincts, they make great companions and are often seen as symbols of mystery and elegance.",
        "Chickens are sociable and resourceful birds commonly raised for their eggs and meat. They have a complex social structure, communicate through various sounds, and can be surprisingly affectionate and friendly when raised around humans."
    ]
    query = "Please describe a cat."

    # Retrieve the best matching document
    best_document = retrieve_best_document(documents, query)

    # Print the result
    print("Best Matching Document:", best_document)
