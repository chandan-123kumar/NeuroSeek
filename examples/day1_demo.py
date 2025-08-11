from storage import NeuroSeekStorage

# Initialize storage
db = NeuroSeekStorage()

# Insert some vectors
db.insert(
    object_id="doc1",
    metadata={"title": "AI in Healthcare"},
    vector=[0.12, 0.45, 0.67, 0.89]
)

db.insert(
    object_id="doc2",
    metadata={"title": "Climate Change Research"},
    vector=[0.91, 0.34, 0.55, 0.12]
)

# Retrieve and print
print("\n[📂] All Stored Objects:")
for oid, data in db.all_objects().items():
    print(f"{oid}: {data}")
