import time
import uuid
from memory.vector_store import VectorStore
from config import TOP_K_RESULTS, SLIDING_WINDOW_SIZE, SYSTEM_PROMPT


class ConversationMemory:
    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.store = VectorStore()
        self.recent_messages: list[dict] = []

    def add_turn(self, role: str, content: str):
        self.recent_messages.append({"role": role, "content": content})
        doc_id = f"{self.session_id}_{int(time.time() * 1000)}_{role}"
        self.store.add(
            doc_id=doc_id,
            text=content,
            metadata={
                "role": role,
                "session_id": self.session_id,
                "timestamp": time.time(),
            },
        )

    def retrieve_context(self, query: str) -> str:
        hits = self.store.query(query, top_k=TOP_K_RESULTS)
        if not hits:
            return ""
        context_parts = []
        for hit in hits:
            role = hit["metadata"].get("role", "unknown")
            context_parts.append(f"[{role}]: {hit['text']}")
        return "\n".join(context_parts)

    def build_prompt(self, user_message: str) -> list[dict]:
        retrieved = self.retrieve_context(user_message)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if retrieved:
            messages.append({
                "role": "system",
                "content": (
                    "Relevant conversation history retrieved via semantic search:\n"
                    f"{retrieved}\n\n"
                    "Use this context to provide coherent, relevant responses."
                ),
            })
        window = self.recent_messages[-SLIDING_WINDOW_SIZE:]
        messages.extend(window)
        messages.append({"role": "user", "content": user_message})
        return messages

    def get_recent(self) -> list[dict]:
        return list(self.recent_messages)

    def clear(self):
        self.recent_messages.clear()
        self.store.clear()
        self.session_id = str(uuid.uuid4())
