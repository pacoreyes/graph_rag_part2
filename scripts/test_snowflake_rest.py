import os
from dotenv import load_dotenv
from curl_cffi import requests

# Load environment variables from .env
load_dotenv()

def test_snowflake_rest_fixed():
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("Error: HUGGING_FACE_HUB_TOKEN not found in .env")
        return

    model_id = "Snowflake/snowflake-arctic-embed-s"
    
    # FIX: Append /pipeline/feature-extraction to force the correct task
    url = f"https://router.huggingface.co/hf-inference/models/{model_id}/pipeline/feature-extraction"
    
    print(f"Testing model: {model_id} via REST API")
    print(f"URL: {url}")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Testing with a list of strings as suggested
    payload = {
        "inputs": ["The new HF router makes model orchestration a breeze."]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            # The result should be a list containing one list (vector)
            if isinstance(result, list) and len(result) > 0:
                embeddings = result[0]
                print("SUCCESS: Retrieved embeddings via REST!")
                print(f"Vector dimensions: {len(embeddings)}")
                print(f"First 5 values: {embeddings[:5]}")
                
                if len(embeddings) == 384:
                    print("Verified: Dimension is 384 as expected.")
            else:
                print(f"Unexpected response format: {result}")
        else:
            print(f"Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_snowflake_rest_fixed()
