from dotenv import load_dotenv
load_dotenv()

from src.impl.datastore import Datastore

store = Datastore()
print("connected, table:", store.table)