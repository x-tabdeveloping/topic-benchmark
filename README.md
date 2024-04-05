# topic-benchmark
Just Benchmarking Topic Models :)

## Timeline:

 - [x] Run benchmark with these models and upload the results:
   - [x] all-MiniLM-L6-v2
   - [x] all-mpnet-base-v2 
   - [x] sentence-transformers/average_word_embeddings_glove.6B.300d 
   - [x] intfloat/e5-large-v2
 - [x] Implement pretty printing and formatting to Latex and MD tables for results.
 - [x] Implement speed tracking.

## Usage:

```bash
pip install topic-benchmark

python3 -m topic_benchmark run -e "embedding_model_name"
```
