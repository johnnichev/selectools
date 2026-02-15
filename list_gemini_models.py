import os

from google import genai

from selectools.env import load_default_env

load_default_env()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("No Gemini API key found.")
    exit(1)

client = genai.Client(api_key=api_key)
try:
    print("Listing models...")
    # The new SDK might have a different way to list models.
    # checking help(client.models) or similar would be good but I can't.
    # I'll try the standard way.
    # client.models.list() ?
    # Or client.models.list_models() ?
    pass
except Exception as e:
    print(f"Error initializing client: {e}")

# Try to list models using standard google.generativeai if installed,
# but we are using google-genai.
# Let's try to infer from the error message "Call ListModels".
# It implies there is a ListModels method.

try:
    for m in client.models.list():
        print(m.name)
except Exception as e:
    print(f"Error listing models: {e}")
