import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables from .env
load_dotenv()

def test_snowflake_embed():
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("Error: HUGGING_FACE_HUB_TOKEN not found in .env")
        return

    model_id = "Snowflake/snowflake-arctic-embed-s"
    print(f"Testing model: {model_id} via InferenceClient")
    
    try:
        client = InferenceClient(api_key=token)
        
        text = "The new HF router makes model orchestration a breeze."
        embeddings = client.feature_extraction(
            text=text,
            model=model_id
        )
        
        print(f"Successfully retrieved embeddings!")
        print(f"Embedding dimensions: {len(embeddings)}")
        print(f"First 5 values: {embeddings[:5]}")
        
        if len(embeddings) == 384:
            print("Verified: Dimension is 384 as expected.")
        else:
            print(f"Warning: Expected 384 dimensions, but got {len(embeddings)}")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_snowflake_embed()
