import requests
from typing import Any
from pydantic import BaseModel
from typing import List, Optional

class AllergyResponse(BaseModel):
    pet_type: str
    allergens: List[str]

def get_pet_allergies(user_input: str) -> AllergyResponse:
    url = "http://localhost:11434/api/chat"  # Replace with your Ollama API endpoint
    headers = {"Content-Type": "application/json"}
    
    # Craft the prompt
    prompt = {
        "model": "llama3.2:3b-instruct-q6_K",  # Replace with your model name
        "messages": [{"role": "user", "content": user_input}],
        "stream": False
    }
    
    # Make the API call
    response = requests.post(url, json=prompt, headers=headers)
    
    # Check for a successful response
    if response.status_code == 200:
        response_data = response.json()
        
        # Assuming the response contains the structured output
        return AllergyResponse(**response_data)
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

# Example usage
try:
    user_input = "My cat has allergies on corn."
    allergies = get_pet_allergies(user_input)
    print(allergies.json())
except Exception as e:
    print(e)