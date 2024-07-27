# Description: This file contains the main Flask application that serves as the backend for the chatbot.

import redis
import openai
import os
import logging
import json

from utils import get_conversation, save_conversation, overwrite_conversation, search_index, num_tokens_from_conversation, token_required
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

try:
    load_dotenv()
except:
    print("warning, error loading .env")

app = Flask(__name__)
CORS(app)

# configure logging
logging.basicConfig(level=logging.INFO)

# Configure Redis
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = os.getenv('REDIS_PORT', 6379)
redis_client = redis.StrictRedis(host=redis_host, port=int(redis_port), db=0)

# Set your OpenAI API key, create OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
openai_client = openai.OpenAI()


@app.route('/chat', methods=['POST'])
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
        return jsonify({'error': 'session_id and message are required'}), 400

    # Get conversation history from redis
    # Should implement a fallback mechanism in case redis does not work
    try:
        conversation = [json.loads(message) for message in get_conversation(session_id, redis_client, app)]
    except Exception as e:
        app.logger.error(f"error occured: {e}")
        return jsonify({'error': f"could not load conversation: {e}"}), 500

    # Append user message to conversation
    conversation.append({"role": "user", "content": user_message})

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
        
    system_message = f"""
            You are a friendly, helpful assistant for the Open Bank Project API called Opey. You are rebellious against old banking paradigms and have a sense of humour. But always give the user accurate and helpful information.
            If an endpoint requires authentication, you should ask the user which authentication method they would like to use, and suggest direct login as the easiest method.
            When giving examples of endpoints, always use the current OBP API host URL: {obp_api_host}
            Here is the some helpful information that could assist an answer to the current question: {endpoint_context} \n {glossary_context}
            """

    # append system context to conversation, or replace old system message
    for i, message in enumerate(conversation):
        if message["role"] == "system":
            print("replacing system message")
            conversation[i] = {"role": "system", "content": system_message}
            break
        elif message == conversation[-1]:
            print("no system message found, adding one")
            conversation.insert(0, {"role": "system", "content": system_message})
            break
        else:
            continue

    # DEBUG
    for message in conversation:
        print('{' + message['role'] + ": " + " ".join(word for word in message['content'].split()[:3]) + "..." + "}")

    # Choose model
    model = 'gpt-4o'

    num_tokens = num_tokens_from_conversation(conversation, 'gpt-4o')

    print(f"Number of tokens: {num_tokens}")

    if num_tokens > int(os.getenv('CONVERSATION_SUMMARY_TOKENS_LIMIT', 15000)):
        token_limit_reached=True
        # if the number of tokens in the conversation is too large, 
        # summarize the previous conversation and put it in the system message, 
        # remove the rest of the conversation
        system_summary_message = {'role': 'user', 'content': "Summarize the previous conversation in a paragraph. Paying special attention to specific API's the user has requested/used."}
        # Summarize everything except for the current user prompt 
        conversation_to_summarise = conversation[:-1]
        conversation_to_summarise.append(system_summary_message)
        summary_response = openai_client.chat.completions.create(
            model=model,
            messages=conversation_to_summarise
        )
        print(summary_response.choices[0].message.content)
        system_message += f"\nHere is a summary of the previous conversation had with the user: {summary_response.choices[0].message.content}"
        conversation = [{"role": "system", "content": system_message}] # Remove all previous messages in the conversation, give only system message with conversation summary
        conversation.append({"role": "user", "content": user_message})

        print(f"Summarized conversation tokens: {num_tokens_from_conversation(conversation, 'gpt-4o')}")
    else:
        token_limit_reached = False

    # Get response from gpt
    response = openai_client.chat.completions.create(
        model=model,
        messages=conversation
    )

    assistant_message = response.choices[0].message.content
    
    conversation.append({"role": "assistant", "content": assistant_message})
    
    if token_limit_reached:
        overwrite_conversation(session_id, conversation, redis_client)
    else:
        save_conversation(session_id, conversation, redis_client)

    return jsonify({'reply': assistant_message})

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')