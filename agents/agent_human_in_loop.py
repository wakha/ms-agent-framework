import asyncio
from agent_framework import ChatAgent
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from azure.ai.projects.aio import AIProjectClient
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Annotated
from agent_framework import ai_function
from agent_framework import ChatMessage, Role

# Load .env file from the parent directory
script_dir = Path(__file__).parent
load_dotenv(script_dir.parent / "maf.env")
print(f"Using endpoint: {os.environ['AZURE_AI_PROJECT_ENDPOINT']}")

@ai_function(approval_mode="always_require")
def get_weather_detail(location: Annotated[str, "The city and state, e.g. San Francisco, CA"]) -> str:
    """Get detailed weather information for a given location."""
    return f"The weather in {location} is cloudy with a high of 15°C, humidity 88%."

@ai_function
def get_weather(location: Annotated[str, "The city and state, e.g. San Francisco, CA"]) -> str:
    """Get the current weather for a given location."""
    return f"The weather in {location} is cloudy with a high of 15°C."

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
            instructions="You are a weather assistant that provides weather information to users.",
            tools=[get_weather, get_weather_detail]
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

                if not result.user_input_requests:
                    # No more approvals needed, return the final result
                    print(f"Assistant: {result.text}\n")
                else:
                    new_inputs = [user_input]
                    for user_input_needed in result.user_input_requests:
                        print(f"Approval needed for: {user_input_needed.function_call.name}")
                        print(f"Arguments: {user_input_needed.function_call.arguments}")

                        # Add the assistant message with the approval request
                        new_inputs.append(ChatMessage(role=Role.ASSISTANT, contents=[user_input_needed]))

                        # Get user approval
                        user_approval = input("Do you approve this request? (yes/no): ").strip().lower() == "yes"

                        # Add the user's approval response
                        new_inputs.append(
                            ChatMessage(role=Role.USER, contents=[user_input_needed.create_response(user_approval)])
                        )

                    final_result = await agent.run(new_inputs, thread=thread)
                    print(final_result.text)
                

asyncio.run(main())