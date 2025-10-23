import asyncio
from agent_framework import ChatAgent
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from azure.ai.projects.aio import AIProjectClient
import os
from pathlib import Path
from dotenv import load_dotenv

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

asyncio.run(main())