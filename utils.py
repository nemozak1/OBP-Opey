# Description: This file contains utility functions used by the API server.
import json
import numpy as np
import faiss
import tiktoken
import redis
import jwt
import flask
import os
import openai
import redis
import json
import aiofiles
import logging

from jwt.exceptions import ExpiredSignatureError, InvalidSignatureError, DecodeError
from functools import wraps
from flask import request, jsonify

logger = logging.getLogger(__name__)

async def get_conversation(session_id, redis_client: redis.StrictRedis, app: flask.app.Flask):
    """
    Retrieve the conversation history from Redis.

    Parameters:
    - session_id (str): The ID of the session for which to retrieve the conversation history.
    - redis_client (redis.StrictRedis): The Redis client used to connect to the Redis server.

    Returns:
    - list: A list of conversation messages, where each message is a string.

    Raises:
    - redis.exceptions.ConnectionError: If there is an error connecting to the Redis server.
    - Exception: If an unexpected error occurs.

    """
    try:
        logger.info(f"Retrieving conversation for session ID: {session_id}")
        conversation = redis_client.lrange(session_id, 0, -1)
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Error connecting to redis server: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        raise
    else:
        return [msg.decode('utf-8') for msg in conversation]


def save_conversation(session_id, conversation, redis_client: redis.StrictRedis):
    """
    Save a conversation to the conversation history in Redis.

    Parameters:
    session_id (str): The ID of the session.
    conversation (list): A list of messages in the conversation.
    redis_client (redis.StrictRedis): The Redis client used to interact with Redis.

    Returns:
    None
    """
    logger.info(f"Saving conversation for session ID: {session_id}")
    # TODO: Need to handle exceptions here
    for message in conversation:
        redis_client.rpush(session_id, json.dumps(message))


def overwrite_conversation(session_id: str, conversation: list, redis_client: redis.StrictRedis):
    """
    Overwrites the conversation for a given session ID in the Redis cache.

    Parameters:
    session_id (str): The ID of the session.
    conversation (list): The list of messages in the conversation.
    redis_client (redis.StrictRedis): The Redis client object.

    Returns:
    None
    """
    try:
        redis_client.delete(session_id)
    except Exception as e:
        print(f"Error overwriting conversation in Redis cache: {e}")

    for message in conversation:
        redis_client.rpush(session_id, json.dumps(message))

async def search_index(query: str, index_file: str, metadata_file: str, client: openai.OpenAI):
    """
    Search a vector database for matches given a query string.

    Args:
        query (str): The query string to search for matches.
        index_file (str): The filename of the Faiss index.
        metadata_file (str): The filename of the metadata JSON file.

    Returns:
        list: A list of matching endpoints.

    Raises:
        FileNotFoundError: If the index or metadata file is not found.
    """
    # Get the query embedding
    embeddings = await get_embeddings([query], client)
    query_embedding = embeddings[0]
    query_embedding_np = np.array([query_embedding]).astype('float32')
    
    # Read the Faiss index
    try:
        index = faiss.read_index(index_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Index file '{index_file}' not found.")
    
    # Perform the search
    distances, indices = index.search(query_embedding_np, k=5)
    
    # Load metadata
    try:
        async with aiofiles.open(metadata_file, 'r') as f:
            contents = await f.read()
        metadata = json.loads(contents)
    except FileNotFoundError:
        raise FileNotFoundError(f"Metadata file '{metadata_file}' not found.")
    
    # Retrieve matching endpoints
    matches = [metadata[i] for i in indices[0]]
    return matches

def num_tokens_from_string(string: str, encoding: tiktoken.Encoding) -> int:
    """Returns the number of tokens for a given string"""
    num_tokens = len(encoding.encode(string))
    return num_tokens

def num_tokens_from_conversation(conversation: list, model: str) -> int:
    """Returns the number of tokens for a given conversation"""
    # Count number of tokens in conversation
    tokens_per_name=1
    tokens_per_message=3
    num_tokens = 0

    # Get encoding for model
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print(f"Tiktoken encoding for model '{model}' not found, defaulting to 'cl100k_base' encoding")
        encoding = tiktoken.get_encoding('cl100k_base')

    for message in conversation:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += num_tokens_from_string(value, encoding)
        if key == "name":
                num_tokens += tokens_per_name

    return num_tokens

# Create vector embeddings
async def get_embeddings(texts, client: openai.OpenAI):
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    
    return [e.embedding for e in response.data]

#Wrapper for JWT required endpoints
def token_required(f):
    """
    Decorator that checks for a valid JSON web token in the Authorization header of the request.
    If a valid token is found, it decodes the token and passes the decoded token as an argument to the decorated function.
    If the token is missing, expired, has an invalid signature, or is invalid, it returns an error response.
    """
    @wraps(f)
    def decorator(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'error': 'Authorization header is missing'}), 401

        token = auth_header.split(" ")[1]
        try:
            public_key = open(os.getenv("OBP_API_EXPLORER_II_PUBLIC_KEY_PATH", "./public_key.pem"), 'r').read()
            decoded_token = jwt.decode(token, public_key, algorithms=["RS256"])
        except ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except InvalidSignatureError:
            return jsonify({'error': 'Invalid signature'}), 401
        except DecodeError:
            return jsonify({'error': 'Invalid token'}), 401

        return f(decoded_token, *args, **kwargs)
   
    return decorator

def verifyJWT(token):
    """
    Verifies the JSON Web Token (JWT) included in the Authorization header of the request.

    Args:
        auth_header (str): The request object containing the Authorization header. i.e. "Bearer <JWT>"

    Returns:
        tuple: A tuple containing the decoded token and a boolean indicating whether the token is valid.
            If the JWT is valid, the decoded token is returned as the first element of the tuple and the boolean value is True.
            If the JWT is invalid or missing, a JSON response with an error message is returned as the first element of the tuple and the boolean value is False.
    """
    
    try:
        public_key = open(os.getenv("OBP_API_EXPLORER_II_PUBLIC_KEY_PATH", "./public_key.pem"), 'r').read()
        decoded_token = jwt.decode(token, public_key, algorithms=["RS256"])
    except ExpiredSignatureError:
        return {'error': 'Token has expired'}, False
    except InvalidSignatureError:
        return {'error': 'Invalid signature'}, False
    except DecodeError:
        return {'error': 'Invalid token'}, False
    except Exception as e:
        return {'error': f'An unknown error occurred: {e}'}, False

    return decoded_token, True