# DocRAG overview

DocRAG is a self-hostable document question-answering system. Documents are uploaded,
split into chunks, embedded locally, and stored in a Qdrant collection.

## Retrieval strategy

Every question runs hybrid retrieval. A dense semantic search and a sparse BM25 keyword
search each return candidates, and the two lists are fused with Reciprocal Rank Fusion
using k=60. Only the fused top candidates are reranked by a cross-encoder, and the
highest-scoring chunks become the context for the answer.

## Embedding models

The default dense embedding model is BAAI/bge-small-en-v1.5 and the default sparse model
is Qdrant/BM25. Embeddings are computed locally and never depend on the generation
provider, so documents can be indexed with no hosted API at all.

## Embedding model tag

The collection stores an embedding model tag. When the configured tag does not match the
tag the collection was built with, DocRAG refuses the query and tells the user to
re-ingest their documents, because vectors from different models are not comparable.
