import google.generativeai as genai
from models.base import BaseModel
from config import GEMINI_API_KEY, FRONTIER_MODEL_NAME


class FrontierModel(BaseModel):
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self._model = genai.GenerativeModel(
            model_name=FRONTIER_MODEL_NAME,
            system_instruction=None,
        )

    @property
    def name(self) -> str:
        return f"Frontier ({FRONTIER_MODEL_NAME})"

    def generate(self, messages: list[dict]) -> str:
        system_parts = []
        chat_history = []
        user_message = ""

        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            elif msg["role"] == "user":
                user_message = msg["content"]
                chat_history.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                chat_history.append({"role": "model", "parts": [msg["content"]]})

        model = genai.GenerativeModel(
            model_name=FRONTIER_MODEL_NAME,
            system_instruction="\n\n".join(system_parts) if system_parts else None,
        )

        history = chat_history[:-1] if chat_history else []
        chat = model.start_chat(history=history)
        response = chat.send_message(user_message)
        return response.text
