from flask import Flask, request, jsonify
from dotenv import load_dotenv
import faiss

import openai
import os
import numpy as np
import json

try:
    load_dotenv()
except:
    print("warning, error loading .env")

app = Flask(__name__)

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI()

def search_index(query, index_name, metadata_name):
    """
    Search a vector database for matches given a query string

    query (str): query string i.e. "what does x endpoint do?"

    index_name (str): filename of faiss index i.e. 'endpoints_index.faiss'

    metadata_name (str): filename of metadata json file (list of texts associated with the index) i.e. 'endpoints_metadata.json' 
    """
    query_embedding = get_embeddings([query])[0]
    query_embedding_np = np.array([query_embedding]).astype('float32')
    
    index = faiss.read_index(index_name)

    # Perform the search
    distances, indices = index.search(query_embedding_np, k=5)
    
    # Load metadata
    with open(metadata_name, 'r') as f:
        metadata = json.load(f)
    
    # Retrieve matching endpoints
    matches = [metadata[i] for i in indices[0]]
    return matches

# Create vector embeddings
def get_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    
    return [e.embedding for e in response.data]

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    endpoint_matches = search_index(user_message, 'endpoints_index.faiss', 'endpoints_metadata.json')
    glossary_matches = search_index(user_message, 'glossary_index.faiss', 'glossary_metadata.json')

    if endpoint_matches:
        endpoint_context = "Here are endpoints (in order of similarity) that matched the users query in a vector database search of the OpenBankProject's API documentation:\n"
        for match in endpoint_matches:
            endpoint_context += f"\nEndpoint: {match['method'].upper()} {match['path']}\n"
            endpoint_context += f"Description: {match['description']}\n"
            endpoint_context += f"Parameters: {', '.join([p['name'] for p in match['parameters']])}\n"
            endpoint_context += f"Responses: {list(match['responses'].keys())}\n"
        
    else:
        endpoint_context = "No relevant endpoints were found for the users query when searching a vector database of the OpenBankProject's API documentation."

        
    if glossary_matches:
        print(glossary_matches)
        glossary_context = "Here are some glossary entries (in order of similarity) that matched the users query in a vector database search of the OpenBankProject's API documentation:\n"
        for match in glossary_matches:
            glossary_context += f"\nTitle: {match['title']}\n"
            glossary_context += f"Description: {match['description']}\n"
        
    else:
        glossary_context = "No relevant glossary items were found for the users query when searching a vector database of the OpenBankProject's API documentation."
        
    system_message = f"""
            You are a helpful assistant for the Open Bank Project API.
            Here is the some helpful information that could assist an answer: {endpoint_context} \n {glossary_context}
            """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    )
    bot_reply = response.choices[0].message.content
    return jsonify({'reply': bot_reply})

if __name__ == '__main__':
    app.run(debug=True)