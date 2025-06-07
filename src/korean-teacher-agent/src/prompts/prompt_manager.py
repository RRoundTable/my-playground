import os
from dotenv import load_dotenv
from phoenix.client import Client
import httpx
from typing import Dict, Optional
import threading
import time
import signal
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Get the DOMAIN value from environment variables
DOMAIN = os.getenv("DOMAIN", "localhost")

# Create a custom httpx client with verify=False to disable SSL verification
transport = httpx.HTTPTransport(verify=False)
http_client = httpx.Client(transport=transport)

client = Client(
    base_url=f"https://phoenix.{DOMAIN}",
    api_key=os.getenv("PHOENIX_API_KEY"),
    # http_client=http_client,
)

class PromptManager:
    """
    A singleton class for managing prompts in Phoenix.
    """
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PromptManager, cls).__new__(cls)
            cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._prompt_cache: Dict[str, object] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._initialized = True
        
    
    def _handle_refresh_signal(self, signum, frame):
        """Handle the refresh signal (SIGUSR1)"""
        print("Received refresh signal, refreshing prompt cache...")
        self.refresh_all_prompts()
    
    
    def get_prompt(self, prompt_name: str, force_refresh: bool = False) -> object:
        """
        Get a prompt from cache or Phoenix if not cached.
        """
        if force_refresh or prompt_name not in self._prompt_cache:
            return self.refresh_prompt(prompt_name)
        return self._prompt_cache[prompt_name]
    
    def get_chat_prompt_template(self, prompt_name: str, variables: dict = {}) -> ChatPromptTemplate:
        """
        Get a chat prompt template from cache or Phoenix if not cached.
        """
        prompt = self.get_prompt(prompt_name)
        formatted_prompt = prompt.format(variables=variables)

        messages =[]
        for message in formatted_prompt.messages:
            messages.append((message["role"], message["content"]))


        return ChatPromptTemplate.from_messages(messages)
   
    
    def refresh_prompt(self, prompt_name: str) -> object:
        """
        Refresh a specific prompt from Phoenix and update cache.
        """
        prompt = client.prompts.get(prompt_identifier=prompt_name)
        self._prompt_cache[prompt_name] = prompt
        self._cache_timestamps[prompt_name] = time.time()
        return prompt
    
    def refresh_all_prompts(self):
        """
        Refresh all cached prompts.
        """
        for prompt_name in list(self._prompt_cache.keys()):
            self.refresh_prompt(prompt_name)
    
    def update_prompt(self, prompt_name: str, prompt: str):
        """
        Update a prompt in Phoenix and refresh cache.
        """
        client.prompts.update(prompt_identifier=prompt_name, prompt=prompt)
        self.refresh_prompt(prompt_name)
    
    def create_prompt(self, prompt_name: str, prompt: str):
        """
        Create a new prompt in Phoenix and cache it.
        """
        created_prompt = client.prompts.create(prompt_identifier=prompt_name, prompt=prompt)
        self._prompt_cache[prompt_name] = created_prompt
        self._cache_timestamps[prompt_name] = time.time()
        return created_prompt
    
    def delete_prompt(self, prompt_name: str):
        """
        Delete a prompt in Phoenix and remove from cache.
        """
        client.prompts.delete(prompt_identifier=prompt_name)
        if prompt_name in self._prompt_cache:
            del self._prompt_cache[prompt_name]
            del self._cache_timestamps[prompt_name]


# Create singleton instance
prompt_manager = PromptManager()

if __name__ == "__main__":
    # Get the prompt from Phoenix (will be cached)
    prompt = prompt_manager.get_chat_prompt_template("writing-homework-idea", variables={
        "format_instructions": "안녕하세요?",
        "request": "안녕하세요?"
    })
    print(prompt)
    import pdb; pdb.set_trace()

 