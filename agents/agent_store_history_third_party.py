import asyncio
from agent_framework import ChatAgent
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from azure.ai.projects.aio import AIProjectClient
import os
from pathlib import Path
from dotenv import load_dotenv
from collections.abc import Sequence
from typing import Any
from uuid import uuid4
from pydantic import BaseModel
import json
import redis.asyncio as redis
from agent_framework import ChatMessage

# Load .env file from the parent directory
script_dir = Path(__file__).parent
load_dotenv(script_dir.parent / "maf.env")
print(f"Using endpoint: {os.environ['AZURE_AI_PROJECT_ENDPOINT']}")

class RedisStoreState(BaseModel):
    """State model for serializing and deserializing Redis chat message store data."""

    thread_id: str
    redis_url: str | None = None
    key_prefix: str = "chat_messages"
    max_messages: int | None = None
    redis_psw: str | None = None


class RedisChatMessageStore:
    """Redis-backed implementation of ChatMessageStore using Redis Lists."""

    def __init__(
        self,
        redis_url: str | None = None,
        thread_id: str | None = None,
        key_prefix: str = "chat_messages",
        max_messages: int | None = None,
        redis_psw: str | None = None

    ) -> None:
        """Initialize the Redis chat message store.

        Args:
            redis_url: Redis connection URL (for example, "redis://localhost:6379").
            thread_id: Unique identifier for this conversation thread.
                      If not provided, a UUID will be auto-generated.
            key_prefix: Prefix for Redis keys to namespace different applications.
            max_messages: Maximum number of messages to retain in Redis.
                         When exceeded, oldest messages are automatically trimmed.
        """
        if redis_url is None:
            raise ValueError("redis_url is required for Redis connection")

        self.redis_url = redis_url
        self.thread_id = thread_id or f"thread_{uuid4()}"
        self.key_prefix = key_prefix
        self.max_messages = max_messages

        # Initialize Redis client
        self._redis_client = redis.Redis(host=redis_url,
                                        port=13635,
                                        decode_responses=True,
                                        username="default",
                                        password=redis_psw,)

    @property
    def redis_key(self) -> str:
        """Get the Redis key for this thread's messages."""
        return f"{self.key_prefix}:{self.thread_id}"

    async def add_messages(self, messages: Sequence[ChatMessage]) -> None:
        """Add messages to the Redis store.

        Args:
            messages: Sequence of ChatMessage objects to add to the store.
        """
        if not messages:
            return

        # Serialize messages and add to Redis list
        serialized_messages = [self._serialize_message(msg) for msg in messages]
        await self._redis_client.rpush(self.redis_key, *serialized_messages)

        # Apply message limit if configured
        if self.max_messages is not None:
            current_count = await self._redis_client.llen(self.redis_key)
            if current_count > self.max_messages:
                # Keep only the most recent max_messages using LTRIM
                await self._redis_client.ltrim(self.redis_key, -self.max_messages, -1)

    async def list_messages(self) -> list[ChatMessage]:
        """Get all messages from the store in chronological order.

        Returns:
            List of ChatMessage objects in chronological order (oldest first).
        """
        # Retrieve all messages from Redis list (oldest to newest)
        redis_messages = await self._redis_client.lrange(self.redis_key, 0, -1)

        messages = []
        for serialized_message in redis_messages:
            message = self._deserialize_message(serialized_message)
            messages.append(message)

        return messages

    async def serialize_state(self, **kwargs: Any) -> Any:
        """Serialize the current store state for persistence.

        Returns:
            Dictionary containing serialized store configuration.
        """
        state = RedisStoreState(
            thread_id=self.thread_id,
            redis_url=self.redis_url,
            key_prefix=self.key_prefix,
            max_messages=self.max_messages,
            redis_psw=self.redis_psw,
        )
        return state.model_dump(**kwargs)

    async def deserialize_state(self, serialized_store_state: Any, **kwargs: Any) -> None:
        """Deserialize state data into this store instance.

        Args:
            serialized_store_state: Previously serialized state data.
            **kwargs: Additional arguments for deserialization.
        """
        if serialized_store_state:
            state = RedisStoreState.model_validate(serialized_store_state, **kwargs)
            self.thread_id = state.thread_id
            self.key_prefix = state.key_prefix
            self.max_messages = state.max_messages

            # Recreate Redis client if the URL changed
            if state.redis_url and state.redis_url != self.redis_url:
                self.redis_url = state.redis_url
                self._redis_client = redis.from_url(self.redis_url, decode_responses=True)

    def _serialize_message(self, message: ChatMessage) -> str:
        """Serialize a ChatMessage to JSON string."""
        message_dict = message.model_dump()
        return json.dumps(message_dict, separators=(",", ":"))

    def _deserialize_message(self, serialized_message: str) -> ChatMessage:
        """Deserialize a JSON string to ChatMessage."""
        message_dict = json.loads(serialized_message)
        return ChatMessage.model_validate(message_dict)

    async def clear(self) -> None:
        """Remove all messages from the store."""
        await self._redis_client.delete(self.redis_key)

    async def aclose(self) -> None:
        """Close the Redis connection."""
        await self._redis_client.aclose()

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
            instructions="You are a helpful assistant that can analyze images and describe what you see.",
             chat_message_store_factory=lambda: RedisChatMessageStore(
                redis_url=os.environ["REDIS_ENDPOINT"],
                max_messages=100,
                redis_psw=os.environ["REDIS_PASSWORD"]
            )
        ) as agent:
            # Create a thread that uses the message store (not service_thread_id)
            thread = agent.get_new_thread(use_message_store=True)
            
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