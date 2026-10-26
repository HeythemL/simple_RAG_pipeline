from typing import List

from interface.base_retriever import BaseRetriever
from impl.datastore import Datastore


class Retriever(BaseRetriever):

    def __init__(self, datastore: Datastore):
        self.datastore = datastore

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        return self.datastore.search(query, top_k=top_k)