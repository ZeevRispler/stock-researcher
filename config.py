import os
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

if not TAVILY_API_KEY or not OPENAI_API_KEY:
    raise ValueError("API keys for Tavily and OpenAI must be set in your .env file.")