# topic-benchmark
Just Benchmarking Topic Models :)

## Todo:

 - [ ] Run benchmark with these models and upload the results:
   - [ ] all-MiniLM-L6-v2 🔁 (currently being rerun)
   - [x] all-mpnet-base-v2 
   - [x] sentence-transformers/average_word_embeddings_glove.6B.300d 
   - [ ] intfloat/e5-large-v2 (OR intfloat/multilingual-e5-large-instruct, to my knowledge, they are the same size, but this one performs way better on MTEB) ⌛
 - [x] Implement pretty printing and formatting to Latex and MD tables for results.
 - [x] Implement speed tracking.

## Usage:

```bash
pip install topic-benchmark

python3 -m topic_benchmark run -e "embedding_model_name"
```
