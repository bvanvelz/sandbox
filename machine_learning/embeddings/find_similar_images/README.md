## Image Embedding
```
usage: image_embedder.py [OPTIONS] FOLDER_PATH
```

## Similarity Search
```
usage: similarity_search.py [-h] [--db-path DB_PATH] [--table-name TABLE_NAME] [--top-k TOP_K] [--open-files]
```

## Demo

### Build LanceDB of image embeddings.
```
python image_embedder.py --table-name images --db-path test/resources/dbs/test1.lance test/resources/images/
```

### Find similar images.
```
python similarity_search.py --table-name images --db-path test/resources/dbs/test1.lance  --top-k 5 --open-files test/resources/images/test_image_1.png
```