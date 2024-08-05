# Description: This file contains the main Flask application that serves as the backend for the chatbot.
import redis
import openai
import os
import logging
import json


from utils import get_conversation, save_conversation, overwrite_conversation, search_index, num_tokens_from_conversation, token_required
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from dotenv import load_dotenv



try:
    load_dotenv()
except:
    print("warning, error loading .env")

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", logger=True)

# configure logging
logging.basicConfig(level=logging.INFO)

# Configure Redis
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = os.getenv('REDIS_PORT', 6379)
app.logger.info(f"Connecting to Redis at {redis_host}:{redis_port}")
redis_client = redis.StrictRedis(host=redis_host, port=int(redis_port), db=0)

# Set your OpenAI API key, create OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
openai_client = openai.OpenAI()

@socketio.on('connect')
def handle_connect(json):
    headers = request.headers
    app.logger.info(f"Connected to the WebSocket: {headers}")
    # Access the headers here and perform any necessary operations
    emit('message', {'data': 'Connected to the WebSocket'})
    return

@socketio.on('chat')
@token_required
def chat(decoded_token):
    """
    Chat function that handles the conversation between the user and the assistant.

    Args:
        decoded_token (dict): Decoded json web token containing user information. This is used to verify incoming requests.
    

    Returns:
        dict: JSON response containing the assistant's reply.
    """
    
    data = request.json
    session_id = data.get('session_id')
    user_message = data.get('message')
    obp_api_host = data.get('obp_api_host')

    app.logger.info(f"Incoming message from user {decoded_token['username']} (obp_user_id: {decoded_token['user_id']}):\n{json.dumps(data, indent=2)}")

    # Validate session_id and user_message
    if not session_id or not user_message:
        emit('error', {'error': 'session_id and message are required'})
        return

    # Get conversation history from redis
    # Should implement a fallback mechanism in case redis does not work
    try:
        conversation = [json.loads(message) for message in get_conversation(session_id, redis_client, app)]
    except Exception as e:
        app.logger.error(f"error occured: {e}")
        emit('error', {'error': f"could not load conversation: {e}"})
        return

    # Append user message to conversation
    conversation.append({"role": "user", "content": user_message})

    # Search for matches in the vector database
    # We query an assistant here to check if additional context is needed to answer the prompt
    thread = openai_client.beta.threads.create()
    context_assistant = openai_client.beta.assistants.retrieve("asst_dtGSW0NS1HbxdpjQAbqXXf9F")
    message = openai_client.beta.threads.messages.create(thread.id, role="user", content=user_message)

    run = openai_client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=context_assistant.id,
    )

    endpoint_matches = search_index(user_message, './endpoint_index.faiss', './endpoint_metadata.json', openai_client)
    glossary_matches = search_index(user_message, './glossary_index.faiss', './glossary_metadata.json', openai_client)

    if endpoint_matches:
        match_list = [f"{m['path']} ({m['summary']})\n" for m in endpoint_matches]
        formatted_matches = ', '.join(match_list)
        app.logger.info(f"Matches for query: \n{formatted_matches}\n")
        endpoint_context = "Here are endpoints (in order of similarity) that matched the users query in a vector database search of the OpenBankProject's API documentation:\n"
        for match in endpoint_matches:
            endpoint_context += f"\nEndpoint: {match['method'].upper()} {match['path']}\n"
            endpoint_context += f"Description: {match['description']}\n"
            if match['parameters'] != {}:
                try:
                    endpoint_context += f"Parameters: {', '.join([p for p in match['parameters']['properties']])}\n"
                except Exception as e:
                    app.logger.error(f"Error obtaining context for {match}: \n{e}")
            else:
                endpoint_context += f"Parameters: This endpoint does not require any parameters"
            responses = [f"{r['code']} {r['body']}" for r in match['responses']]
            formatted_responses = ', '.join(responses)
            endpoint_context += f"Responses: \n{formatted_responses}\n"
            
        
    else:
        endpoint_context = "No relevant endpoints were found for the users query when searching a vector database of the OpenBankProject's API documentation."

        
    if glossary_matches:
        glossary_context = "Here are some glossary entries (in order of similarity) that matched the users query in a vector database search of the OpenBankProject's API documentation:\n"
        for match in glossary_matches:
            glossary_context += f"\nTitle: {match['title']}\n"
            glossary_context += f"Description: {match['description']}\n"
        
    else:
        glossary_context = "No relevant glossary items were found for the users query when searching a vector database of the OpenBankProject's API documentation."
        

    # Retrieve opey assistant from OpenAI
    opey = openai_client.beta.assistants.retrieve("asst_vbwdYbWsTisP7YmwQhykiEwp")
    # Create a run with the user message
    with openai_client.beta.threads.runs.stream(
        thread_id=thread.id,
        assistant_id=opey.id,
        additional_instructions=f"""When giving examples of endpoints, always use the current OBP API host URL: {obp_api_host}
            Here is the some helpful information that could assist an answer to the current question: {endpoint_context} \n {glossary_context}"""
    ) as stream: 
        for event in stream:
            # Print the text from text delta events
            if event.event == "thread.message.delta" and event.data.delta.content:
                print(event.data.delta.content[0].text)
                emit('message', {'data': event.data.delta.content[0].text})

    

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0')