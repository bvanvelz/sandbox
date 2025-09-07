import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import lancedb
from datetime import datetime
import typer

class ImageEmbedder:
    def __init__(self, db_path: str):

        """
        Initialize the ImageEmbedder with CLIP model and LanceDB connection.
        """
        self.db_path = db_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load CLIP model for image embeddings
        print(f"Loading OpenCLIP model on {self.device}...")

        # Old model: "openai/clip-vit-base-patch32" (512d embeddings)
        # self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # New model: OpenCLIP ViT-L/14 (768d embeddings, better performance)
        self.model = CLIPModel.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
        self.processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")

        self.model.to(self.device)
        
        # Connect to LanceDB
        self.db = lancedb.connect(db_path)
        
        # Supported image extensions
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        print("✅ ImageEmbedder initialized successfully!")
    
    def is_image_file(self, file_path: str) -> bool:
        """Check if a file is an image based on its extension."""
        return Path(file_path).suffix.lower() in self.image_extensions
    
    def walk_directory(self, folder_path: str) -> List[str]:
        """Walk through directory and find all image files."""
        image_files = []
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            raise ValueError(f"Directory {folder_path} does not exist")
        
        print(f"Scanning directory: {folder_path}")
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = Path(root) / file
                if self.is_image_file(str(file_path)):
                    image_files.append(str(file_path))
        
        print(f"Found {len(image_files)} image files")
        return image_files
    
    def get_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """Extract metadata from an image file."""
        try:
            with Image.open(image_path) as img:
                # Get basic image info
                width, height = img.size
                format_name = img.format
                mode = img.mode
                
                # Get file info
                file_path = Path(image_path)
                file_size = file_path.stat().st_size
                file_name = file_path.name
                file_extension = file_path.suffix.lower()
                
                # Generate file hash for deduplication
                with open(image_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                
                # Get creation/modification times
                stat = file_path.stat()
                created_time = datetime.fromtimestamp(stat.st_ctime).isoformat()
                modified_time = datetime.fromtimestamp(stat.st_mtime).isoformat()
                
                return {
                    "file_path": str(image_path),
                    "file_name": file_name,
                    "file_extension": file_extension,
                    "file_size": file_size,
                    "file_hash": file_hash,
                    "width": width,
                    "height": height,
                    "format": format_name,
                    "mode": mode,
                    "created_time": created_time,
                    "modified_time": modified_time,
                    "processed_time": datetime.now().isoformat()
                }
        except Exception as e:
            print(f"Error processing metadata for {image_path}: {e}")
            return {}
    
    def generate_embedding(self, image_path: str) -> np.ndarray:
        """Generate embedding for an image using CLIP model."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            
            # Process image with CLIP
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Move inputs to device
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    inputs[key] = value.to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.model.get_image_features(**inputs)
            
            # Convert to numpy array and normalize
            embedding = embedding.cpu().numpy().flatten()
            
            return embedding
            
        except Exception as e:
            print(f"Error generating embedding for {image_path}: {e}")
            return np.array([])
    
    def process_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Process a list of images to generate embeddings and metadata."""
        results = []
        total_images = len(image_paths)
        
        print(f"Processing {total_images} images...")
        
        for i, image_path in enumerate(image_paths):
            try:
                print(f"Processing {i+1}/{total_images}: {Path(image_path).name}")
                
                # Get metadata
                metadata = self.get_image_metadata(image_path)
                if not metadata:
                    continue
                
                # Generate embedding
                embedding = self.generate_embedding(image_path)
                if len(embedding) == 0:
                    continue
                
                # Ensure embedding is float32
                embedding = np.asarray(embedding, dtype=np.float32)
                
                # Combine metadata and embedding
                result = {
                    **metadata,
                    "embedding": embedding.tolist(),
                    "embedding_dim": len(embedding)
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        print(f"Successfully processed {len(results)} out of {total_images} images")
        return results
    
    def create_table(self, table_name: str = "images") -> None:
        """Create a new table in LanceDB for storing image embeddings."""
        try:
            # Check if table exists
            if table_name in self.db.table_names():
                print(f"Table '{table_name}' already exists")
                return
            # Create table with sample data to establish schema
            import numpy as np
            sample_data = [{
                "file_path": "sample",
                "file_name": "sample", 
                "file_extension": "sample",
                "file_size": 0,
                "file_hash": "sample",
                "width": 0,
                "height": 0,
                "format": "sample",
                "mode": "sample",
                "created_time": "sample",
                "modified_time": "sample",
                "processed_time": "sample",
                "embedding": np.zeros(768, dtype=np.float32).tolist(),  # Updated to 768d for new model
                "embedding_dim": 768  # Updated to 768d for new model
            }]
            table = self.db.create_table(table_name, data=sample_data)
            print(f"Created table '{table_name}'")
            # Delete the sample row
            table.delete(where="file_path == 'sample'")
        except Exception as e:
            print(f"Error creating table: {e}")
    
    def insert_embeddings(self, data: List[Dict[str, Any]], table_name: str = "images") -> None:
        """Insert embeddings and metadata into LanceDB table, skipping duplicates by file_hash."""
        try:
            if not data:
                print("No data to insert")
                return
            # Get or create table
            if table_name not in self.db.table_names():
                self.create_table(table_name)
            table = self.db.open_table(table_name)
            # Fetch existing file_hashes for deduplication
            existing_hashes = set()
            try:
                existing_records = table.search().select(["file_hash"]).to_pandas()
                existing_hashes = set(existing_records["file_hash"].tolist())
            except Exception as e:
                print(f"Warning: Could not fetch existing hashes for deduplication: {e}")
            # Filter out duplicates
            unique_data = []
            for row in data:
                import numpy as np
                emb = np.asarray(row["embedding"], dtype=np.float32)
                if emb.shape[0] != 768:  # Updated to 768d for new model
                    raise ValueError(f"Embedding has wrong shape: {emb.shape}")
                row["embedding"] = emb.tolist()
                if row["file_hash"] not in existing_hashes:
                    unique_data.append(row)
                else:
                    print(f"Skipping duplicate image: {row['file_name']} (hash: {row['file_hash']})")
            if not unique_data:
                print("No unique data to insert (all duplicates)")
                return
            # Insert unique data
            table.add(unique_data)
            print(f"Successfully inserted {len(unique_data)} new records into table '{table_name}' (skipped {len(data) - len(unique_data)} duplicates)")
        except Exception as e:
            print(f"Error inserting data: {e}")
    
    def process_folder(self, folder_path: str, table_name: str = "images") -> None:
        """Complete pipeline: walk directory, generate embeddings, and store in LanceDB."""
        print(f"🚀 Starting image processing pipeline for: {folder_path}")
        
        # Step 1: Walk directory and find image files
        print("\n📁 Step 1: Scanning directory for images...")
        image_files = self.walk_directory(folder_path)

        if not image_files:
            print("No image files found!")
            return
        
        # Step 2: Generate embeddings and metadata
        print("\n🔍 Step 2: Generating embeddings...")
        processed_data = self.process_images(image_files)
        
        if not processed_data:
            print("No images were successfully processed!")
            return
        
        # Step 3: Insert into LanceDB
        print("\n💾 Step 3: Storing in LanceDB...")
        self.insert_embeddings(processed_data, table_name)
        
        print(f"\n✅ Pipeline completed! Processed {len(processed_data)} images")
        print(f"Database location: {self.db_path}")
        print(f"Table name: {table_name}")

def main(
    table_name: str = typer.Option(..., "--table-name", help="Name of the LanceDB table."),
    db_path: str = typer.Option(..., "--db-path", help="Path to LanceDB database."),
    folder_path: str = typer.Argument(..., help="Path to folder containing images.")
):
    """Process images and store embeddings in LanceDB."""
    # Initialize embedder
    embedder = ImageEmbedder(db_path)
    
    # Process folder
    embedder.process_folder(folder_path, table_name)

if __name__ == "__main__":
    typer.run(main) 