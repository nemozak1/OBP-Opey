import requests
import openai
import logging
import os
import faiss
import json
import numpy as np

from dotenv import load_dotenv
from markdownify import markdownify

load_dotenv()

# Set up logging
logging_level = os.getenv('LOG_LEVEL', 'INFO')
logger = logging.basicConfig(level=logging_level)

client = openai.OpenAI()

# Directory where to save vector index
VECTOR_DIRECTORY = "./vector-database"

# Get documentation from OBP
obp_base_url = "https://apisandbox.openbankproject.com"
obp_version = "v5.1.0"

# If the swagger path is overriden by an environment variable, use that
obp_swagger_url_from_env = os.getenv('OBP_SWAGGER_URL')
if obp_swagger_url_from_env:
    logging.info(f"Swagger URL Overriden from environment variable OBP_SWAGGER_URL: {obp_swagger_url_from_env}")
    swagger_url = f"{obp_base_url}{obp_swagger_url_from_env}"
else:
    swagger_url = "{}/obp/v5.1.0/resource-docs/{}/swagger?locale=en_GB".format(obp_base_url, obp_version)
    logging.info(f"Swagger URL not overriden, using default: {swagger_url}")

# Get the _static_ swagger docs, we may want to change this if we give this to a bank that has lots of dynamic endpoints

logging.info(f"Requesting swagger docs from {swagger_url}")
try:
    swagger_response = requests.get(swagger_url)
except Exception as e:
    logging.error(f"Error fetching swagger docs: {e}")
    logging.error(e.traceback.format_exc())

if swagger_response.status_code != 200:
    logging.error(f"Error fetching swagger docs: {swagger_response.text}")
    logging.info("If the swagger endpoint is broken, try overriding it with a known working one by setting OBP_SWAGGER_URL\n NOT RECOMMENDED FOR PRODUCTION")
    raise Exception(f"Error fetching swagger docs: {swagger_response.text}")
else:
    logging.info("Swagger docs fetched successfully")

swagger_json = swagger_response.json()


# get the glossary from OBP
glossary_url = "{}/obp/{}/api/glossary".format(obp_base_url, obp_version)
logging.info(f"Requesting glossary from {glossary_url}")
try:
    glossary_response = requests.get(glossary_url)

except Exception as e:
    logging.error(f"Error fetching glossary: {e}")
    logging.error(e.traceback.format_exc())

if glossary_response.status_code != 200:
    logging.error(f"Error fetching glossary: {glossary_response.text}")
    raise Exception(f"Error fetching glossary: {glossary_response.text}")
else:
    logging.info("Glossary fetched successfully")

glossary_json = glossary_response.json()


def resolve_reference(ref, definitions, resolved={}):
    """
    Resolves a $ref to its definition, avoiding circular references.

    Parameters:
    ref (str): The reference to be resolved.
    definitions (dict): A dictionary containing the definitions.
    resolved (dict, optional): A dictionary containing the resolved references. Defaults to an empty dictionary.

    Returns:
    dict: The resolved definition.

    """
    ref_name = ref.split('/')[-1]
    if ref_name in resolved:
        return resolved[ref_name]

    if ref_name in definitions.keys():
        definition = definitions[ref_name]
    else:
        definition = {}
    resolved[ref_name] = definition
    properties = definition.get('properties', {})
    resolved_properties = resolve_properties(properties, definitions, resolved)
    return {**definition, 'properties': resolved_properties}


def resolve_properties(properties, definitions, resolved):
    """
    Resolves nested references in properties, avoiding circular references.
    """
    resolved_properties = {}
    for prop_name, prop_details in properties.items():
        if '$ref' in prop_details:
            resolved_properties[prop_name] = resolve_reference(prop_details['$ref'], definitions, resolved)
        elif prop_details.get('type') == 'array' and 'items' in prop_details and '$ref' in prop_details['items']:
            resolved_properties[prop_name] = {
                "type": "array",
                "items": resolve_reference(prop_details['items']['$ref'], definitions, resolved)
            }
        else:
            resolved_properties[prop_name] = prop_details
    return resolved_properties

def parse_swagger(swagger_json):
    """
    Parses a Swagger JSON file and extracts information about the endpoints.

    Args:
        swagger_json (dict): The Swagger JSON object.

    Returns:
        list: A list of dictionaries, where each dictionary represents an endpoint and its details.
    """
    try:
        paths = swagger_json['paths']
    except KeyError:
        raise KeyError("no 'paths' key found in swagger JSON, ")
    definitions = swagger_json['definitions']

    endpoints = []
    for path, methods in paths.items():
        for method, details in methods.items():
            endpoint_info = {
                'path': path,
                'method': method,
                'summary': details.get('summary'),
                'description': markdownify(details.get('description', '')),
                'responses': [],
                'parameters': {}
            }

            if ('parameters' in details) & (details['parameters'] != []):
                endpoint_info["parameters"] = {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    }
                for param in details['parameters']:
                    if param['in'] == 'body' and '$ref' in param['schema']:
                        ref = param['schema']['$ref']
                        definition = resolve_reference(ref, definitions)
                        
                        endpoint_info['parameters']['required'].extend(definition.get('required', []))
                        endpoint_info['parameters']['properties'].update(
                            resolve_properties(definition.get('properties', {}), definitions, {})
                        )
                    elif param['in'] == 'body' and '$ref' not in param['schema']:
                        # Right now, if the parameter does not have a reference (i.e. something that points to a swagger definition) we skip it
                        endpoint_info["parameters"] = param["schema"]
                        
                    elif param['in'] == 'path':
                        endpoint_info['parameters']['required'].append(param['name'])
                        endpoint_info['parameters']['properties'][param['name']] = {
                            "type": param['type'],
                            "in": "path",
                            "description": param.get('description', '')
                        }
                    elif param['in'] == 'query':
                        endpoint_info['parameters']['properties'][param['name']] = {
                            "type": param['type'],
                            "description": param.get('description', '')
                        }
                        if param.get('required', False):
                            endpoint_info['parameters']['required'].append(param['name'])
            
            if 'responses' in details:
                for code, response in details['responses'].items():
                    if "schema" in response.keys() and ("$ref" in response['schema']):
                        ref = response['schema']['$ref']
                        definition_name = ref.split('/')[-1]
                        definition = resolve_reference(ref, definitions)

                        response_resolved = {
                            "code": code,
                            "body": resolve_properties(definition.get('properties', {}), definitions, {})
                        }

                        endpoint_info["responses"].append(response_resolved)
                        
            endpoints.append(endpoint_info)
    return endpoints

endpoints = parse_swagger(swagger_json)

def parse_glossary(glossary_json):
    """
    Parses the glossary JSON and extracts the title and description of each glossary item.
    
    Args:
        glossary_json (dict): The glossary JSON containing the glossary items.
        
    Returns:
        list: A list of dictionaries, where each dictionary represents a parsed glossary item
              with 'title' and 'description' keys.
    """
    
    glossary_items = glossary_json['glossary_items']
    parsed_items = []
    
    for item in glossary_items:
        title = item.get('title', 'No title')
        description_info = item.get('description', {})
        
        # Get markdown description or else return no description
        description = description_info.get('markdown', 'No description')
        # do not add descriptions if they are empty
        if description == "":
            continue
        
        parsed_items.append({
            'title': title,
            'description': description
        })
    
    return parsed_items

glossary_items = parse_glossary(glossary_json)

# Create vector embeddings
def get_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    
    return [e.embedding for e in response.data]

def create_and_save_embedding_faiss(formatted_texts: list, json_metadata: list, filename: str):
    """
    Creates and saves text embeddings and metadata for a given list of texts.

    Args:
        formatted_texts (list): A formatted list of texts for creating embeddings.
        json_metadata (list): A list of dictionaries to pass as JSON metadata. Each dictionary represents metadata for a text.
        filename (str): A prefix to attach to the saved index and metadata files.

    Returns:
        None

    Raises:
        None

    Example usage:
        create_and_save_embedding_faiss(formatted_texts, json_metadata, filename)
    """
    embeddings = get_embeddings(formatted_texts)
    
    # Convert embeddings to a numpy array
    embeddings_np = np.array(embeddings).astype('float32')
    
    # Create a FAISS index
    index = faiss.IndexFlatL2(embeddings_np.shape[1])  # L2 distance index
    index.add(embeddings_np)
    
    # Optionally, save the index to disk for later use
    faiss.write_index(index, f"{filename}_index.faiss")
    
    # Save metadata for retrieval
    with open(f"{filename}_metadata.json", 'w') as f:
        json.dump(json_metadata, f)



glossary_texts = [f"{g['title']} - {g['description']}" for g in glossary_items]
if not os.path.isdir(VECTOR_DIRECTORY):
    try:
        os.mkdir(VECTOR_DIRECTORY)
    except Exception as err:
        print(f"Error creating directory {VECTOR_DIRECTORY}: {err}")


create_and_save_embedding_faiss(glossary_texts, glossary_items, os.path.join(VECTOR_DIRECTORY, "glossary"))

endpoint_texts = [f"{e['method'].upper()} {e['path']} - {e['description']}" for e in endpoints]
create_and_save_embedding_faiss(endpoint_texts, endpoints, os.path.join(VECTOR_DIRECTORY, "endpoint"))
