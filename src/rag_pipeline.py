from typing import List

from interface.base_datastore import BaseDatastore
from interface.base_indexer import BaseIndexer
from interface.base_response_generator import BaseResponseGenerator
from interface.base_retriever import BaseRetriever


class RAGPipeline:
    """
    Central orchestrator that wires together all pipeline components.
    Only depends on interfaces - concrete implementations are injected.
    """

    def __init__(
        self,
        datastore: BaseDatastore,
        indexer: BaseIndexer,
        retriever: BaseRetriever,
        response_generator: BaseResponseGenerator,
        top_k: int = 3,
    ):
        self.datastore = datastore
        self.indexer = indexer
        self.retriever = retriever
        self.response_generator = response_generator
        self.top_k = top_k

    def reset(self) -> None:
        """Drop and recreate the vector store."""
        print("[*] Resetting datastore...")
        self.datastore.reset()
        print("[OK] Datastore reset complete.")

    def add_documents(self, document_paths: List[str]) -> None:
        """
        Index documents and store their chunks in the datastore.

        Args:
            document_paths: list of file paths to index (e.g. PDFs).
        """
        print(f"[*] Indexing {len(document_paths)} document(s)...")
        items = self.indexer.index(document_paths)
        print(f"    -> {len(items)} chunk(s) created. Adding to datastore...")
        self.datastore.add_items(items)
        print("[OK] Documents indexed and stored.")

    def process_query(self, query: str) -> str:
        """
        Retrieve relevant chunks and generate a grounded answer.

        Args:
            query: the user's question.

        Returns:
            Generated answer string.
        """
        context: List[str] = self.retriever.retrieve(query, top_k=self.top_k)
        return self.response_generator.generate_response(query, context)
