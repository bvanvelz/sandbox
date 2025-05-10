import os
import torch
import faiss
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

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

# 4. Find similar images
def find_similar_images(query_path, image_paths, image_embeddings, k=5):
    query_embedding = embed_image(query_path)
    index = faiss.IndexFlatL2(image_embeddings.shape[1])
    index.add(image_embeddings)
    distances, indices = index.search(query_embedding, k)
    return [image_paths[i] for i in indices[0]]

# Example usage
if __name__ == "__main__":
    import numpy as np

    image_folder = "./test_images"   # Folder with .jpg/.png files
    query_image = "./query.png"                # Path to query image

    # Embed dataset
    paths, embs = embed_images_from_folder(image_folder)

    # Find similar
    similar_images = find_similar_images(query_image, paths, embs)
    print("Top similar images:")
    for sim in similar_images:
        print(sim)