import asyncio
from concurrent.futures import thread
from agent_framework import ChatAgent
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from azure.ai.projects.aio import AIProjectClient
import os
from pathlib import Path
from dotenv import load_dotenv
import json
import tempfile

# Load .env file from the parent directory
script_dir = Path(__file__).parent
load_dotenv(script_dir.parent / "maf.env")
print(f"Using endpoint: {os.environ['AZURE_AI_PROJECT_ENDPOINT']}")

async def main():
    async with AzureCliCredential() as credential:
        # Create the project client from the endpoint
        project_client = AIProjectClient(
            endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"],
            credential=credential
        )
        
        # Create the agent with the project client
        async with ChatAgent(
            chat_client=AzureAIAgentClient(
                project_client=project_client, 
                model_deployment_name=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
                async_credential=credential
            ),
            instructions="You are a helpful assistant that can analyze images and describe what you see."
        ) as agent:

            temp_dir = os.getcwd()#tempfile.gettempdir()
            file_path = os.path.join(temp_dir, "agent_thread.json")

            if os.path.exists(file_path):
                # File exists, load it
                print("Loading existing thread from file...")
                with open(file_path, "r") as f:
                    loaded_json = f.read()
                reloaded_data = json.loads(loaded_json)
                thread = await agent.deserialize_thread(reloaded_data)
            else:
                # File doesn't exist, create new thread
                thread = agent.get_new_thread()
            
            print("Chat started! Type 'end' to exit.\n")
            
            while True:
                user_input = input("You: ")
                
                if user_input.lower() == "end":
                    print("Goodbye!")
                    break
                
                if not user_input.strip():
                    continue
                
                result = await agent.run(user_input, thread=thread)
                print(f"Assistant: {result.text}\n")

            # Serialize the thread state
            serialized_thread = await thread.serialize()
            serialized_json = json.dumps(serialized_thread)

            # Example: save to a local file (replace with DB or blob storage in production)
            with open(file_path, "w") as f:
                f.write(serialized_json)

asyncio.run(main())