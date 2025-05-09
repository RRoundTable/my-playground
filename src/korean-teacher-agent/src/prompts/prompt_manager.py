import os
from dotenv import load_dotenv
from phoenix.client import Client
import httpx

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

def get_prompt(prompt_name: str) -> str:
    """
    Get a prompt from Phoenix.
    """
    prompt = client.prompts.get(prompt_identifier=prompt_name)
    return prompt

def update_prompt(prompt_name: str, prompt: str):
    """
    Update a prompt in Phoenix.
    """
    client.prompts.update(prompt_identifier=prompt_name, prompt=prompt)

def create_prompt(prompt_name: str, prompt: str):
    """
    Create a new prompt in Phoenix.
    """
    client.prompts.create(prompt_identifier=prompt_name, prompt=prompt)

def delete_prompt(prompt_name: str):
    """
    Delete a prompt in Phoenix.
    """
    client.prompts.delete(prompt_identifier=prompt_name)




if __name__ == "__main__":
    # Get the prompt from Phoenix
    prompt = get_prompt("korean-teacher-agent")
    
    # Print the prompt ID
    print(f"Prompt ID: {prompt.id}")
    
    # Use the _loads method to parse the prompt data
    # This method takes a PromptVersionData or PromptVersion object and returns a parsed instance
    parsed_prompt = prompt._dumps()
    import pprint
    pprint.pprint(parsed_prompt)
   
    formatted_prompt = prompt.format(variables={"question": "안녕하세요?"})
    print(formatted_prompt)