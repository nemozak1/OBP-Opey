# Description: This file contains the framework for the chatbot itself.

import logging
import json
import traceback
import importlib

obp = importlib.import_module("obp-python-apiv5.1")

from utils import get_conversation, verifyJWT, search_index
from openai import AsyncAssistantEventHandler
from typing_extensions import override
from langchain.agents.openai_assistant import OpenAIAssistantRunnable

class Opey():

    async def fetch_assistant(self, assistant_id):
        """
        Get the assistant from openAI as langchain agent

        Parameters:
        assistant_id (str): The ID of the openai assistant used for Opey.
        """
        self.agent = await OpenAIAssistantRunnable(assistant_id=assistant_id, as_agent=True)
        return

    async def 


class Conversation():
    def __init__(self, assistant_id: str):
        """
        Initializes a new instance of the Conversation class.

        Parameters:
        assistant_id (str): The ID of the openai assistant used for Opey.

        Returns:
        None
        """
        self.assistant_id = assistant_id
        self.conversation = []

    async def handle_connect(self, user_sid, auth):
        # Access the headers here and perform any necessary operations
        response, jwt_valid = verifyJWT(auth["token"])
        if not jwt_valid:
            await sio.emit('error', {'error': 'Invalid JWT'}, to=user_sid)
            logging.error(f"Invalid JWT: {response}")
            await sio.disconnect()
            return
        else:
            logging.info(f"WebSocket opened with: {response}")
            logging.info("Connecting to assistant")
            self.opey_assistant = await openai_client.beta.assistants.retrieve(self.assistant_id)
            logging.info("Creating new assistant thread")
            self.thread = await openai_client.beta.threads.create()
            await sio.emit('message', {'data': 'Chatting with Opey'}, to=user_sid)
        return

    async def handle_chat(self, user_sid, data):
        """
        Chat function that handles the conversation between the user and the assistant.

        Args:
            decoded_token (dict): Decoded json web token containing user information. This is used to verify incoming requests.
        

        Returns:
            dict: JSON response containing the assistant's reply.
        """
        
        session_id = data.get('session_id')
        user_message = data.get('message')
        obp_api_host = data.get('obp_api_host')

        #print(f"Incoming message from user {decoded_token['username']} (obp_user_id: {decoded_token['user_id']}):\n{json.dumps(data, indent=2)}")

        # Validate session_id and user_message
        if not session_id or not user_message:
            await sio.emit('error', {'error': 'session_id and message are required'}, to=user_sid)
            return

        # Add message to assistant thread
        message = await openai_client.beta.threads.messages.create(self.thread.id, role="user", content=user_message)

        # Get conversation history from redis
        # Should implement a fallback mechanism in case redis does not work
        try:
            conversation = [json.loads(message) for message in await get_conversation(session_id, redis_client, app)]
        except Exception as e:
            logging.error(f"error occurred: {str(e)}")
            logging.error(traceback.format_exc())  # Add this line to print the traceback
            await sio.emit('error', {'error': f"could not load conversation"}, to=user_sid)
            return

        # Append user message to conversation
        conversation.append({"role": "user", "content": user_message})

        # Search for matches in the vector database
        # We query an assistant here to check if additional context is needed to answer the prompt
        context_classifier_assistant = await openai_client.beta.assistants.retrieve("asst_dtGSW0NS1HbxdpjQAbqXXf9F")

        context_run = await openai_client.beta.threads.runs.create_and_poll(
            thread_id=self.thread.id,
            assistant_id=context_classifier_assistant.id,
        )

        if context_run.status == "completed":
            messages = await openai_client.beta.threads.messages.list(thread_id=self.thread.id)
            context_messages = [msg async for msg in messages if msg.assistant_id == context_classifier_assistant.id]
            
        try:
            result = json.loads(context_messages[0].content[0].text.value) 
            logging.info(f"Context requirements: {result['context_required']}")
        except Exception as e:
            logging.info(f"Could not get context requirements from assistant: {e}")

        if result['context_required'] == 'true':
            endpoint_matches = await search_index(user_message, './endpoint_index.faiss', './endpoint_metadata.json', openai_client)
            glossary_matches = await search_index(user_message, './glossary_index.faiss', './glossary_metadata.json', openai_client)

            if endpoint_matches:
                match_list = [f"{m['path']} ({m['summary']})\n" for m in endpoint_matches]
                formatted_matches = ', '.join(match_list)
                logging.info(f"Matches for query: \n{formatted_matches}\n")
                endpoint_context = "Here are endpoints (in order of similarity) that matched the users query in a vector database search of the OpenBankProject's API documentation:\n"
                for match in endpoint_matches:
                    endpoint_context += f"\nEndpoint: {match['method'].upper()} {match['path']}\n"
                    endpoint_context += f"Description: {match['description']}\n"
                    if match['parameters'] != {}:
                        try:
                            endpoint_context += f"Parameters: {', '.join([p for p in match['parameters']['properties']])}\n"
                        except Exception as e:
                            logging.info(f"Error obtaining context for {match}: \n{e}")
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
        else:
            endpoint_context = "The user message did not require additional endpoint context to answer the prompt."
            glossary_context = "The user message did not require additional glossary context to answer the prompt."

        # Create a run with the user message
        async with openai_client.beta.threads.runs.stream(
            thread_id=self.thread.id,
            assistant_id=self.opey_assistant.id,
            additional_instructions=f"""When giving examples of endpoints, always use the current OBP API host URL: {obp_api_host}
                Here is the some helpful information that could assist an answer to the current question: {endpoint_context} \n {glossary_context}""",
            event_handler=OpeyEventHandler(user_sid=user_sid),
            tool_choice="none", # This is to prevent context being added from the file_search tool as well as the vector searching we are doing in house, which causes the rate limit to be reached frequently
        ) as stream:
            await stream.until_done()


class OpeyEventHandler(AsyncAssistantEventHandler):
    def __init__(self, user_sid, *args, **kwargs):
        self.user_sid = user_sid
        super().__init__(*args, **kwargs)

    @override
    async def on_text_delta(self, delta, snapshot):
        await sio.emit('response stream delta', {'assistant': f"{delta.value}"}, to=self.user_sid) 
    
    async def on_message_created(self, message):
        await sio.emit('response stream start', to=self.user_sid)

    async def on_event(self, event) -> None:

        if event.event != "thread.message.delta":
            logging.info(f"Event: {event.event}")

        if event.event == "thread.run.failed":
            logging.error(f"Thread run failed: {event.data.last_error}")
            if event.data.last_error.code == "rate_limit_exceeded":
                await sio.emit('error', {'error': 'Rate limit exceeded'}, to=self.user_sid)
            else:
                await sio.emit('error', {'error': 'Thread run failed'}, to=self.user_sid)

        return super().on_event(event)
    
    async def on_message_done(self, message):
        await sio.emit('response stream end', to=self.user_sid)

