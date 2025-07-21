#!/usr/bin/env python3
"""
Query script for the LanceDB image embeddings database.
This script demonstrates various ways to query the database.
"""

import lancedb
import numpy as np
from pathlib import Path

def query_database(db_path, table_name):
    """Query the LanceDB database with various examples."""
    
    # Connect to database
    db = lancedb.connect(db_path)
    print(f"📊 Database: {db_path}")
    print(f"📋 Tables: {db.table_names()}")
    
    # Open the images table
    table = db.open_table(table_name)
    print(f"\n📈 Table '{table_name}' has {table.count_rows()} rows")
    print(f"🔍 Schema: {table.schema}")
    
    # Get first 10 records and print
    print("\n" + "="*50)
    print("FIRST 10 RECORDS")
    print("="*50)
    first_10 = table.to_pandas().head(10)
    for i, record in first_10.iterrows():
        print(f"\nRecord {i+1}:")
        print(f"  File: {record['file_name']}")
        print(f"  Path: {record['file_path']}")
        print(f"  Size: {record['file_size']} bytes")
        print(f"  Dimensions: {record['width']}x{record['height']}")
        print(f"  Format: {record['format']}")
        print(f"  Embedding dim: {record['embedding_dim']}")

    
    '''
    # Example 1: Get all records
    print("\n" + "="*50)
    print("EXAMPLE 1: All Records")
    print("="*50)
    all_records = table.to_pandas()
    print(f"Total records: {len(all_records)}")
    
    for i, record in all_records.iterrows():
        print(f"\nRecord {i+1}:")
        print(f"  File: {record['file_name']}")
        print(f"  Path: {record['file_path']}")
        print(f"  Size: {record['file_size']} bytes")
        print(f"  Dimensions: {record['width']}x{record['height']}")
        print(f"  Format: {record['format']}")
        print(f"  Embedding dim: {record['embedding_dim']}")

    # Example 2: Filter by file extension
    print("\n" + "="*50)
    print("EXAMPLE 2: PNG Files Only")
    print("="*50)
    png_files = table.search().where("file_extension = '.png'").to_pandas()
    print(f"Found {len(png_files)} PNG files:")
    for _, record in png_files.iterrows():
        print(f"  - {record['file_name']}")
    
    # Example 3: Get metadata without embeddings
    print("\n" + "="*50)
    print("EXAMPLE 3: Metadata Only (No Embeddings)")
    print("="*50)
    metadata = table.search().select([
        "file_name", "file_path", "width", "height", 
        "file_size", "format", "created_time"
    ]).to_pandas()
    
    print("Metadata summary:")
    for _, record in metadata.iterrows():
        print(f"  {record['file_name']}: {record['width']}x{record['height']} ({record['format']})")
    
    # Example 4: Find largest files
    print("\n" + "="*50)
    print("EXAMPLE 4: Largest Files")
    print("="*50)
    # Use pandas to sort by file_size
    largest_files = all_records.sort_values("file_size", ascending=False).head(3)
    print("Top 3 largest files:")
    for i, (_, record) in enumerate(largest_files.iterrows()):
        size_kb = record['file_size'] / 1024
        print(f"  {i+1}. {record['file_name']}: {size_kb:.1f} KB")
    
    # Example 5: Search for specific file
    print("\n" + "="*50)
    print("EXAMPLE 5: Search for Specific File")
    print("="*50)
    test_image = table.search().where("file_name = 'test_image_1.png'").to_pandas()
    if len(test_image) > 0:
        record = test_image.iloc[0]
        print(f"Found: {record['file_name']}")
        print(f"  Path: {record['file_path']}")
        print(f"  Size: {record['file_size']} bytes")
        print(f"  Dimensions: {record['width']}x{record['height']}")
        print(f"  Embedding length: {len(record['embedding'])}")
    else:
        print("File not found")
    
    # Example 6: Get embedding statistics
    print("\n" + "="*50)
    print("EXAMPLE 6: Embedding Statistics")
    print("="*50)
    embeddings = table.search().select(["embedding_dim", "file_name"]).to_pandas()
    print("Embedding dimensions:")
    for _, record in embeddings.iterrows():
        print(f"  {record['file_name']}: {record['embedding_dim']} dimensions")
    
    # Example 7: Complex query - files created today
    print("\n" + "="*50)
    print("EXAMPLE 7: Files Created Today")
    print("="*50)
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    today_files = table.search().where(f"created_time LIKE '{today}%'").to_pandas()
    print(f"Files created today: {len(today_files)}")
    for _, record in today_files.iterrows():
        print(f"  - {record['file_name']}")
    '''

if __name__ == "__main__":
    # Query the database
    query_database(db_path="dbs/image_embeddings.lance", table_name="images")
    
    print("\n✅ Database query completed!") 