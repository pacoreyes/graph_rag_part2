import os
from dotenv import load_dotenv
from curl_cffi import requests

# Load environment variables from .env
load_dotenv()

def test_snowflake_standard_api():
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("Error: HUGGING_FACE_HUB_TOKEN not found in .env")
        return

    model_id = "Snowflake/snowflake-arctic-embed-s"
    # Standard Inference API URL
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    
    print(f"Testing model: {model_id} via Standard API ({url})")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Standard embedding payload
    payload = {
        "inputs": "The new HF router makes model orchestration a breeze."
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            embeddings = response.json()
            print("Successfully retrieved embeddings via Standard API!")
            print(f"Result type: {type(embeddings)}")
            if isinstance(embeddings, list):
                print(f"Embedding dimensions: {len(embeddings)}")
                print(f"First 5 values: {embeddings[:5]}")
        else:
            print(f"Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_snowflake_standard_api()
