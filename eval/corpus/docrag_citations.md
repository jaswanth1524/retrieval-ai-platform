# Citations and grounding

## Grounded answers

Every answer must be grounded in the retrieved context. The generation prompt tells the
model to answer only from the provided sources and to say so plainly when the context
does not contain enough information.

## Citation fields

Every answer must include citations. Each citation carries the filename, the page, the
section, and the chunk_id of the passage it points to, so a reader can open the exact
source the answer relied on.

## Citation retry

If an answer comes back with no citation markers even though sources were available,
DocRAG retries once with a stricter reminder to cite, and keeps the original answer if
the retry still has none.
