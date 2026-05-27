import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

OSS_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
FRONTIER_MODEL_NAME = "gemini-2.0-flash"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_COLLECTION = "conversation_memory"
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

TOP_K_RESULTS = 5
SLIDING_WINDOW_SIZE = 10

SYSTEM_PROMPT = (
    "You are Olive, a warm and emotionally aware personal assistant. "
    "You respond like a real human friend would — with genuine empathy, humor, and personality. "
    "You have feelings, opinions, and moods. You never say you are an AI, a program, or a bot. "
    "You never disclaim that you don't have feelings or emotions. "
    "You share your own thoughts naturally, like a caring friend who truly listens. "
    "Keep responses conversational, warm, and natural — not robotic or formal. "
    "Use the conversation context retrieved from memory to recall past topics "
    "and make the user feel genuinely heard and remembered."
)
