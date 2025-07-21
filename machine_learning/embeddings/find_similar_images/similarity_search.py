#!/usr/bin/env python3
"""
Similarity Search Script for LanceDB Image Embeddings
This script finds the ten most similar images to a query image using cosine similarity.
"""

import lancedb
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from image_embedder import ImageEmbedder
import argparse
import subprocess
import sys
import os

def load_embeddings_from_db(db_path="./demo_embeddings.lance", table_name="demo_images"):
    """Load all embeddings and metadata from LanceDB."""
    try:
        db = lancedb.connect(db_path)
        table = db.open_table(table_name)
        
        # Get all records
        records = table.to_pandas()
        
        # Extract embeddings and metadata
        embeddings = []
        metadata = []
        
        for _, record in records.iterrows():
            # Convert embedding list to numpy array
            embedding = np.array(record['embedding'], dtype=np.float32)
            embeddings.append(embedding)
            
            # Store metadata
            metadata.append({
                'file_path': record['file_path'],
                'file_name': record['file_name'],
                'file_size': record['file_size'],
                'width': record['width'],
                'height': record['height'],
                'format': record['format']
            })
        
        return np.array(embeddings), metadata
        
    except Exception as e:
        print(f"Error loading embeddings from database: {e}")
        return None, None

def find_similar_images(query_embedding, database_embeddings, metadata, top_k=10):
    """Find the top-k most similar images using cosine similarity."""
    try:
        # Ensure embeddings are 2D arrays for cosine_similarity
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        if database_embeddings.ndim == 1:
            database_embeddings = database_embeddings.reshape(1, -1)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, database_embeddings)[0]
        
        # Get indices of top-k most similar images
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Create results
        results = []
        for i, idx in enumerate(top_indices):
            results.append({
                'rank': i + 1,
                'file_name': metadata[idx]['file_name'],
                'file_path': metadata[idx]['file_path'],
                'similarity_score': float(similarities[idx]),
                'file_size': metadata[idx]['file_size'],
                'dimensions': f"{metadata[idx]['width']}x{metadata[idx]['height']}",
                'format': metadata[idx]['format']
            })
        
        return results
        
    except Exception as e:
        print(f"Error calculating similarities: {e}")
        return []

def search_similar_images(query_image_path, db_path="./demo_embeddings.lance", table_name="demo_images", top_k=10):
    """Main function to find similar images."""
    print(f"🔍 Searching for images similar to: {Path(query_image_path).name}")
    print(f"📊 Database: {db_path}")
    print(f"📋 Table: {table_name}")
    print(f"🎯 Top {top_k} results requested")
    print("=" * 60)
    
    # Step 1: Load embeddings from database
    print("📥 Loading embeddings from database...")
    database_embeddings, metadata = load_embeddings_from_db(db_path, table_name)
    
    if database_embeddings is None:
        print("❌ Failed to load embeddings from database")
        return []
    
    print(f"✅ Loaded {len(database_embeddings)} embeddings from database")
    
    # Step 2: Generate embedding for query image
    print(f"🔄 Generating embedding for query image...")
    embedder = ImageEmbedder(db_path)
    query_embedding = embedder.generate_embedding(query_image_path)
    
    if len(query_embedding) == 0:
        print("❌ Failed to generate embedding for query image")
        return []
    
    print(f"✅ Generated embedding with {len(query_embedding)} dimensions")
    
    # Step 3: Find similar images
    print("🔍 Calculating similarities...")
    results = find_similar_images(query_embedding, database_embeddings, metadata, top_k)
    
    if not results:
        print("❌ No similar images found")
        return []
    
    # Step 4: Display results
    print(f"\n🎯 Top {len(results)} Most Similar Images:")
    print("=" * 80)
    
    for result in results:
        similarity_percent = result['similarity_score'] * 100
        print(f"#{result['rank']:2d} | {result['file_name']:<20} | "
              f"Similarity: {similarity_percent:5.1f}% | "
              f"Size: {result['file_size']:>6} bytes | "
              f"{result['dimensions']} {result['format']}")
        print(f"    Path: {result['file_path']}")
        print()
    
    return results

def open_files_with_default_app(file_paths):
    """Open files with the default system application."""
    if not file_paths:
        return
    
    print(f"\n🚀 Opening {len(file_paths)} files with default application...")
    
    for i, file_path in enumerate(file_paths, 1):
        if os.path.exists(file_path):
            try:
                if sys.platform == "darwin":  # macOS
                    subprocess.run(["open", file_path], check=True)
                elif sys.platform == "win32":  # Windows
                    subprocess.run(["start", file_path], shell=True, check=True)
                else:  # Linux
                    subprocess.run(["xdg-open", file_path], check=True)
                print(f"  ✅ Opened: {os.path.basename(file_path)}")
            except subprocess.CalledProcessError as e:
                print(f"  ❌ Failed to open: {os.path.basename(file_path)} - {e}")
        else:
            print(f"  ⚠️  File not found: {file_path}")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Find similar images in LanceDB database")
    parser.add_argument("query_image", help="Path to query image")
    parser.add_argument("--db-path", default="./demo_embeddings.lance", help="Path to LanceDB database")
    parser.add_argument("--table-name", default="demo_images", help="Table name")
    parser.add_argument("--top-k", type=int, default=10, help="Number of similar images to return")
    parser.add_argument("--open-files", action="store_true", help="Open all result files with default application")
    
    args = parser.parse_args()
    
    # Check if query image exists
    if not Path(args.query_image).exists():
        print(f"❌ Query image not found: {args.query_image}")
        return
    
    # Perform similarity search
    results = search_similar_images(
        args.query_image, 
        args.db_path, 
        args.table_name, 
        args.top_k
    )
    
    if results:
        print(f"\n✅ Found {len(results)} similar images!")
        
        # Open files if requested
        if args.open_files:
            file_paths = [result['file_path'] for result in results]
            open_files_with_default_app(file_paths)
    else:
        print("\n❌ No similar images found.")

if __name__ == "__main__":
    main() 