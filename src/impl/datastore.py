import lancedb
from lancedb.table import Table
import pyarrow as pa
from typing import List
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from interface.base_datastore import BaseDatastore, DataItem

# all-MiniLM-L6-v2: fast, lightweight, 384-dim, runs fully locally
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_DIMENSIONS = 384


class Datastore(BaseDatastore):
    DB_PATH = "data/sample-lancedb"
    DB_TABLE_NAME = "rag-table"

    def __init__(self):
        self.vector_dimensions = VECTOR_DIMENSIONS
        print("[*] Loading embedding model (first run may download it)...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print("[OK] Embedding model ready.")
        self.vector_db = lancedb.connect(self.DB_PATH)
        self.table = self._get_table()

    def reset(self) -> Table:
        # Drop the table if it exists
        try:
            self.vector_db.drop_table(self.DB_TABLE_NAME)
        except Exception:
            print("[INFO] Unable to drop table. Assuming it doesn't exist.")

        # Create the new table.
        schema = pa.schema(
            [
                pa.field("vector", pa.list_(pa.float32(), self.vector_dimensions)),
                pa.field("content", pa.utf8()),
                pa.field("source", pa.utf8()),
            ]
        )

        self.table = self.vector_db.create_table(self.DB_TABLE_NAME, schema=schema)
        print(f"[OK] Table Reset/Created: {self.DB_TABLE_NAME} in {self.DB_PATH}")
        return self.table

    def get_vector(self, content: str) -> List[float]:
        embedding = self.embedding_model.encode(content, show_progress_bar=False)
        return embedding.tolist()

    def add_items(self, items: List[DataItem]) -> None:
        # Encode all chunks in one batched call (much faster than one-by-one)
        contents = [item.content for item in items]
        print(f"[*] Encoding {len(contents)} chunks...")
        vectors = self.embedding_model.encode(contents, show_progress_bar=True, batch_size=64)

        entries = [
            {"vector": vec.tolist(), "content": item.content, "source": item.source}
            for item, vec in zip(items, vectors)
        ]

        self.table.merge_insert(
            "source"
        ).when_matched_update_all().when_not_matched_insert_all().execute(entries)

    def search(self, query: str, top_k: int = 5) -> List[str]:
        vector = self.get_vector(query)
        results = (
            self.table.search(vector)
            .select(["content", "source"])
            .limit(top_k)
            .to_list()
        )
        return [row["content"] for row in results]

    def _convert_item_to_entry(self, item: DataItem) -> dict:
        return {
            "vector": self.get_vector(item.content),
            "content": item.content,
            "source": item.source,
        }

    def _get_table(self) -> Table:
        try:
            return self.vector_db.open_table(self.DB_TABLE_NAME)
        except Exception:
            print("[INFO] Table not found. Creating a new one...")
            return self.reset()