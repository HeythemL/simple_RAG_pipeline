from abc import ABC, abstractmethod
from typing import List


class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        pass