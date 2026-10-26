import sys
sys.path.insert(0, 'src')
from dotenv import load_dotenv
load_dotenv()
from src.impl.datastore import Datastore
from src.impl.retriever import Retriever

ds = Datastore()
rows = ds.table.to_pandas()
print(f"Total rows in table: {len(rows)}")

jed_rows = rows[rows['content'].str.contains('Jed', case=False, na=False)]
print(f"Rows mentioning 'Jed': {len(jed_rows)}")
if len(jed_rows) > 0:
    print("\nSample chunk with 'Jed':")
    print(jed_rows['content'].iloc[0][:400])

print("\n--- Top 3 retrieved for 'Who founded Swan Lagoon?' ---")
r = Retriever(ds)
chunks = r.retrieve("Who founded Swan Lagoon?", top_k=3)
for i, c in enumerate(chunks):
    safe = c.encode('ascii', errors='replace').decode('ascii')
    print(f"[{i+1}] {safe[:300]}\n")
