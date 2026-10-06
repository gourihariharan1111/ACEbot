
from dotenv import load_dotenv
from google import genai
import os

# Load variables from .env
load_dotenv()

# Get Gemini API key
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not found.")

print("API key found.")

# Create Gemini client
client = genai.Client(api_key=api_key)

print("Sending request to Gemini...")

try:
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents="Say hello in one short sentence."
    )

    print("\nGemini response:")
    print(response.text)

except Exception as e:
    print("\nGemini request failed:")
    print(type(e).__name__)
    print(e)

