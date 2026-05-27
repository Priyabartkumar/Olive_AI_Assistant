from huggingface_hub import InferenceClient
from models.base import BaseModel
from config import HF_API_TOKEN, OSS_MODEL_NAME


class OSSModel(BaseModel):
    def __init__(self):
        self._client = InferenceClient(
            model=OSS_MODEL_NAME,
            token=HF_API_TOKEN,
        )

    @property
    def name(self) -> str:
        return f"OSS ({OSS_MODEL_NAME.split('/')[-1]})"

    def generate(self, messages: list[dict]) -> str:
        response = self._client.chat_completion(
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
        )
        return response.choices[0].message.content
