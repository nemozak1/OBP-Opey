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
Note: If you are running this on your local docker engine and you already have an instance of redis running, you may need to change the ```REDIS_PORT``` in the env file to avoid clashing
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
    * For running locally, set ```REDIS_HOST=localhost```
    * **You will also need redis set up and running locally**, find instructions for this [here](https://redis.io/docs/latest/operate/oss_and_stack/install/install-redis/)

#### 3. Create vector index
We need to register the OBP API documentation in a vector index, run:
```bash
python create_vector_index.py
```
#### 4. Run
For development:
```
flask --app './main.py' run
```
For production we use gunicorn:
```
gunicorn --bind 0.0.0.0:5000 main:app
```
#### 5. Chat
Same as for docker (see above)
