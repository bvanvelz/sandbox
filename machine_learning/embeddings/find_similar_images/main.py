import os
import torch
import faiss
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

# config
INDEX_PATH = './index.faiss'


# 1. Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 2. Preprocess and embed image
def embed_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    return embeddings.cpu().numpy()

# 3. Embed a folder of images
def embed_images_from_folder(folder):
    paths = []
    embeddings = []
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, filename)
            print(f"Generating embedding for path '{path}'.")
            emb = embed_image(path)
            paths.append(path)
            embeddings.append(emb)

    return paths, np.vstack(embeddings)

def add_images_from_folder(folder):
    paths = []
    embeddings = []
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, filename)
            print(f"Generating embedding for path '{path}'.")
            emb = embed_image(path)
            paths.append(path)
            embeddings.append(emb)

    image_embeddings = np.vstack(embeddings)

    return paths, 

    index = faiss.IndexFlatL2(image_embeddings.shape[1])

    index.add(image_embeddings)

    

    # Persist index.
    print("Persisting index.")
    faiss.write_index(index, INDEX_PATH)


# 4. Find similar images
def find_similar_images(query_path, image_paths, index, k=10):
    query_embedding = embed_image(query_path)

    distances, indices = index.search(query_embedding, k)

    print(f"Distances from query: {distances}")

    return [image_paths[i] for i in indices[0]]


def get_index():
    """
    Loads faiss database from disk.
    """
    return 
    if os.isfile(INDEX_PATH):
        faiss.read_index(INDEX_PATH) 
    else:


# Example usage
if __name__ == "__main__":
    import numpy as np

    # image_folder = "./test_images"   # Folder with .jpg/.png files
    image_folder = ""  # FLAG
    query_image = "./query.jpg"      # Path to query image

    # Embed dataset
    paths, embs = embed_images_from_folder(image_folder)

    index = get_index(embs)

    # Find similar
    '''
    similar_images = find_similar_images(query_image, paths, index)
    print("Top similar images:")
    for sim in similar_images:
        print(sim)
    '''

    print("Done.")