import asyncio
import os
from dotenv import load_dotenv
from agent.infrastructure.hf_embedding_client import HFEmbeddingClient

# Load environment variables from .env
load_dotenv()

async def test_hf_client_integration():
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("Error: HUGGING_FACE_HUB_TOKEN not found in .env")
        return

    model_name = "Snowflake/snowflake-arctic-embed-s"
    print(f"Testing HFEmbeddingClient integration with model: {model_name}")
    
    client = HFEmbeddingClient(model_name=model_name, api_token=token)
    
    try:
        text = "This is an integration test for the HFEmbeddingClient."
        vector = await client.embed(text)
        
        print("SUCCESS: Retrieved embedding vector!")
        print(f"Vector dimensions: {len(vector)}")
        print(f"First 5 values: {vector[:5]}")
        
        if len(vector) == 384:
            print("Verified: Dimension is 384 as expected.")
        else:
            print(f"Warning: Expected 384 dimensions, but got {len(vector)}")
            
    except Exception as e:
        print(f"Integration test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_hf_client_integration())
