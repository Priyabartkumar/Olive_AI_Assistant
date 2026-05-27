from abc import ABC, abstractmethod


class BaseModel(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def generate(self, messages: list[dict]) -> str: ...
