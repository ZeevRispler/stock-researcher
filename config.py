import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Fetch API keys and optional base URL from the environment
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")  # For proxies or local models

# Simple validation to ensure required API keys are set
if not TAVILY_API_KEY or not OPENAI_API_KEY:
    raise ValueError(
        "API keys for Tavily and OpenAI must be set in your .env file."
    )