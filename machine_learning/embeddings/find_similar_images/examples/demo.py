#!/usr/bin/env python3
"""
Demo script for the Image Embedder application.
This script demonstrates how to use the ImageEmbedder class to process images
and store embeddings in LanceDB.
"""

import os
from pathlib import Path
from image_embedder import ImageEmbedder

def create_sample_images():
    """Create a sample folder with some test images for demonstration."""
    sample_dir = Path("./sample_images")
    sample_dir.mkdir(exist_ok=True)
    
    # Create a simple test image using PIL
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    
    # Create different colored squares as test images
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    
    for i, color in enumerate(colors):
        # Create a 200x200 image with the color
        img = Image.new('RGB', (200, 200), color=color)
        
        # Add some text to make them different
        draw = ImageDraw.Draw(img)
        draw.text((50, 90), f"Image {i+1}", fill='white')
        
        # Save the image
        img_path = sample_dir / f"test_image_{i+1}.png"
        img.save(img_path)
        print(f"Created test image: {img_path}")
    
    return str(sample_dir)

def demo_image_embedding():
    """Demonstrate the complete image embedding pipeline."""
    print("🚀 Image Embedding Demo")
    print("=" * 50)
    
    # Step 1: Create sample images
    print("\n📁 Step 1: Creating sample images...")
    sample_folder = create_sample_images()
    
    # Step 2: Initialize the embedder
    print("\n🔧 Step 2: Initializing ImageEmbedder...")
    embedder = ImageEmbedder("./demo_embeddings.lance")
    
    # Step 3: Process the folder
    print("\n🔄 Step 3: Processing images...")
    embedder.process_folder(sample_folder, "demo_images")
    
    # Step 4: Demonstrate similarity search
    print("\n🔍 Step 4: Demonstrating similarity search...")
    sample_images = list(Path(sample_folder).glob("*.png"))
    if sample_images:
        query_image = str(sample_images[0])
        print(f"Searching for images similar to: {Path(query_image).name}")
        
        similar_images = embedder.search_similar_images(query_image, "demo_images", top_k=3)
        
        print("\nSimilar images found:")
        for i, result in enumerate(similar_images):
            print(f"{i+1}. {Path(result['file_path']).name} (Score: {result.get('_distance', 'N/A')})")
    
    print("\n✅ Demo completed!")
    print(f"Database location: ./demo_embeddings.lance")
    print(f"Table name: demo_images")

def demo_command_line_usage():
    """Show how to use the command line interface."""
    print("\n📋 Command Line Usage:")
    print("=" * 50)
    print("To process a folder of images:")
    print("python image_embedder.py /path/to/your/images")
    print()
    print("With custom options:")
    print("python image_embedder.py /path/to/your/images --table-name my_images --db-path ./my_db.lance")
    print()
    print("Example:")
    print("python image_embedder.py ./sample_images --table-name test_images")

if __name__ == "__main__":
    # Run the demo
    demo_image_embedding()
    
    # Show command line usage
    demo_command_line_usage() 