from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

import faiss
import redis
import openai
import os
import numpy as np
import json

try:
    load_dotenv()
except:
    print("warning, error loading .env")

app = Flask(__name__)
CORS(app)

# Configure Redis
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# Set your OpenAI API key, create OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

def get_conversation(session_id):
    """Retrieve the conversation history from Redis."""
    conversation = redis_client.lrange(session_id, 0, -1)
    print(conversation)
    return [msg.decode('utf-8') for msg in conversation]

def save_conversation(session_id, conversation):
    """Save a message to the conversation history in Redis."""
    for message in conversation:
        redis_client.rpush(session_id, json.dumps(message))

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
    data = request.json
    session_id = data.get('session_id')
    user_message = data.get('message')
    obp_api_host = data.get('obp_api_host')

    # Get conversation history from redis
    conversation = [json.loads(message) for message in get_conversation(session_id)]

    # Append user message to conversation
    conversation.append({"role": "user", "content": user_message})

    endpoint_matches = search_index(user_message, './Data/endpoints_index.faiss', './Data/endpoints_metadata.json')
    glossary_matches = search_index(user_message, './Data/glossary_index.faiss', './Data/glossary_metadata.json')

    if endpoint_matches:
        print(endpoint_matches)
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
            You are a friendly, helpful assistant for the Open Bank Project API called Opey. You are rebellious against old banking paradigms and have a sense of humour. But always give the user accurate and helpful information.
            When giving examples of endpoints, always use the current OBP API host URL: {obp_api_host}
            Here is the some helpful information that could assist an answer to the current question: {endpoint_context} \n {glossary_context}
            """

    # append system context to conversation, or replace old system message
    for message in conversation:
        if message["role"] == "system":
            print("replacing system message")
            message = {"role": "system", "content": system_message}
        elif message == conversation[-1]:
            print("no system message found, adding one")
            conversation.append({"role": "system", "content": system_message})
        else:
            continue
    

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=conversation
    )

    assistant_message = response.choices[0].message.content
    
    conversation.append({"role": "assistant", "content": assistant_message})
    
    save_conversation(session_id, conversation)

    return jsonify({'reply': assistant_message})

if __name__ == '__main__':
    app.run(debug=True)