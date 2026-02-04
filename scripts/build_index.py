import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    # Load the world rules
    with open('data/world_rules.json', 'r') as f:
        world = json.load(f)
        
    chunks = []
    
    # Character mappings
    for name, info in world['characters'].items():
        text = f"{name}: {info['mapping']}"
        if 'role' in info:
            text += f". Role: {info['role']}"
        if 'status' in info:
            text += f". Status: {info['status']}"
        if 'problem' in info:
            text += f". Problem: {info['problem']}"
        chunks.append(text)
    
    # Object mappings
    for obj, info in world['objects'].items():
        text = f"{obj}: {info['mapping']}"
        if 'what_it_does' in info:
            text += f". {info['what_it_does']}"
        if 'meaning' in info:
            text += f". {info['meaning']}"
        if 'validity' in info:
            text += f". Validity: {info['validity']}"
        chunks.append(text)
    
    # Concepts
    for concept, meaning in world['concepts'].items():
        chunks.append(f"{concept} means {meaning}")
    
    print(f"Building index with {len(chunks)} chunks...")
    
    # Generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, convert_to_numpy=True)
    
    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    faiss.write_index(index, "data/world_vectors.index")
    np.save("data/world_chunks.npy", np.array(chunks))
    
    print(f"Done. Index: {index.ntotal} vectors, dim {dimension}")

if __name__ == "__main__":
    main()