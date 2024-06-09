# OBP-GPT (Opey)
A helpful chatbot trained on Open Bank Project API documentation

### Running with Docker (reccommended)
#### 1. Set up Environment
* [Install](https://docs.docker.com/compose/install/) docker compose if you haven't already
* Copy .env-example to .env
    * Copy over your OpenAI API Key

#### 2. Build App
```bash
sudo docker compose build
```
#### 3. Run
```bash
sudo docker compose up
```
#### 4. Chat
The chat endpoint should now be running at http://127.0.0.1:5000/chat

The best way to interact with Opey locally is by running a version of [Api Explorer II](https://github.com/nemozak1/API-Explorer-II) Locally and using the chat widget.

Else you can chat with the bot using curl (or whatever http client you like):

```curl
curl -XPOST -H "Content-type: application/json" -d '{
  "session_id": "123456789", "obp_api_host": "https://test.openbankproject.com", "message": "Which endpoint would I use to create a new customer at a bank?"
}' 'http://127.0.0.1:5000/chat'
```
### Running Locally
#### 1. Install dependencies
```bash
pip install -r requirements.txt
```
#### 2. Set up Environment
* Copy .env-example to .env
    * Copy over your OpenAI API Key

#### 3. Create vector index
We need to register the OBP API documentation in a vector index, run:
```bash
python create_vector_index.py
```
#### 4. Chat
Same as for docker (see above)
