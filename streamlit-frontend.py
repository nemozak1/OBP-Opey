import streamlit as st
import requests
import json
import websocket
import threading
import os
# Backend server URL
backend_url = "ws://localhost:5000"

# Initialize WebSocket connection
ws = None

# Create a lock object to synchronize access to the WebSocket
ws_lock = threading.Lock()

# Function to connect to the WebSocket server
def connect_to_server():
    global ws
    with ws_lock:
        ws = websocket.WebSocketApp(backend_url,
                                    on_message=on_message,
                                    on_error=on_error,
                                    on_close=on_close)
        ws.on_open = on_open
        ws.run_forever()

# WebSocket event handlers
def on_message(ws, message):
    st.session_state["messages"].append({"role": "assistant", "content": message})
    st.experimental_rerun()

def on_error(ws, error):
    st.session_state["messages"].append({"role": "error", "content": str(error)})
    st.experimental_rerun()

def on_close(ws, close_status_code, close_msg):
    st.session_state["messages"].append({"role": "system", "content": "Connection closed"})
    st.experimental_rerun()

def on_open(ws):
    st.session_state["messages"].append({"role": "system", "content": "Connected to server"})
    st.experimental_rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# UI components
st.title("Chat with Opey Assistant")

# Show chat history
for message in st.session_state["messages"]:
    if message["role"] == "user":
        st.write(f"**You:** {message['content']}")
    elif message["role"] == "assistant":
        st.write(f"**Opey:** {message['content']}")
    elif message["role"] == "error":
        st.error(f"Error: {message['content']}")
    elif message["role"] == "system":
        st.info(message["content"])

# Input form for user message
user_input = st.text_input("Your message:", key="input")

# Send message on form submission
if st.button("Send"):
    if ws:
        with ws_lock:
            ws.send(json.dumps({"session_id": "test_session", "message": user_input, "obp_api_host": "https://api.openbankproject.com"}))
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.experimental_rerun()
    else:
        st.error("WebSocket connection is not open")

# Connect to the server in a separate thread
if ws is None:
    threading.Thread(target=connect_to_server, daemon=True).start()

# This will ensure the Streamlit app only works when Flask is run with `--debug`
if __name__ == "__main__":
    if "DEBUG" in os.environ:
        st.warning("Debug mode is on. The app is running in an unsecured mode for testing.")
    else:
        st.error("The app is not in debug mode. Please start the Flask app with `--debug` to enable the Streamlit UI.")
