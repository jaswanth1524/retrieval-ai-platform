# Settings reference

<!-- Generated from api/settings.py by `make settings-docs`; edit the comments there. -->

Every setting is an environment variable (or a line in `.env`): the name below,
upper-cased. `.env.example` lists them all at their defaults. Settings marked
*per workspace* can also be changed for one workspace in Settings → Administration.

- [Vector store](#vector-store)
- [Embedding and reranking models](#embedding-and-reranking-models)
- [Retrieval and ranking](#retrieval-and-ranking)
- [Generation providers](#generation-providers)
- [Answers and conversations](#answers-and-conversations)
- [Summaries](#summaries)
- [Documents and chunking](#documents-and-chunking)
- [Indexing jobs](#indexing-jobs)
- [Originals, backups and workspaces](#originals-backups-and-workspaces)
- [Watched folders](#watched-folders)
- [Web sources](#web-sources)
- [Access and limits](#access-and-limits)
- [Logs, traces and feedback](#logs-traces-and-feedback)
- [Notifications](#notifications)

## Vector store

Where Qdrant is and how the collection is laid out.

- **`QDRANT_URL`** (text; default `http://localhost:6333`) — Qdrant's REST address (Compose sets http://qdrant:6333).
- **`QDRANT_COLLECTION`** (text; default `docrag_documents`) — The default workspace's collection; other workspaces add __ws_&lt;name&gt;.
- **`QDRANT_DENSE_VECTOR_NAME`** (text; default `dense`) — Name of the dense vector on each point (part of the collection schema).
- **`QDRANT_SPARSE_VECTOR_NAME`** (text; default `sparse`) — Name of the sparse (BM25) vector on each point (part of the collection schema).
- **`QDRANT_DENSE_VECTOR_SIZE`** (integer; default `384`; > 0) — Dimensions of DENSE_EMBEDDING_MODEL's vectors (bge-small: 384).
- **`QDRANT_PREFER_GRPC`** (true/false; default false) — Talk to Qdrant over gRPC instead of REST. gRPC avoids per-call JSON (de)serialization overhead versus Qdrant's REST API — meaningful at query volume, free to enable since the client already exposes both transports over the same connection details.
- **`QDRANT_GRPC_PORT`** (integer; default `6334`; ≥ 1, ≤ 65535) — The gRPC port the client dials when qdrant_prefer_grpc is on (the host it uses is qdrant_url's). Compose also maps it on the host, and pins it to 6334 for the API container, which reaches Qdrant inside the compose network.
- **`QDRANT_API_KEY`** (text; default empty) — Qdrant's own API key (the server's QDRANT__SERVICE__API_KEY). Empty (default) sends none. Without it anyone who can reach Qdrant's port reads or drops the collection directly, around API_KEY entirely.
- **`QDRANT_QUANTIZATION`** ("none" or "scalar"; default `none`) — Dense vectors kept as int8 in RAM beside the float32 originals ("scalar"), about a quarter of the memory for a large corpus; the top candidates are re-scored against the originals (oversampled by QDRANT_QUANTIZATION_OVERSAMPLING), so ranking stays close. "none" (default): float32 only. Applied to an existing collection in place (no re-embedding); switching back removes it.
- **`QDRANT_QUANTIZATION_OVERSAMPLING`** (number; default `2.0`; ≥ 1.0, ≤ 8.0) — With QDRANT_QUANTIZATION=scalar: how many times more candidates are re-scored against the full vectors.

## Embedding and reranking models

The local models. Changing an embedding setting needs a re-index.

- **`DENSE_EMBEDDING_MODEL`** (text; default `BAAI/bge-small-en-v1.5`) — Local FastEmbed dense model. Changing it is a new vector space: bump EMBEDDING_MODEL_TAG and re-index (questions are refused until the tag matches).
- **`SPARSE_EMBEDDING_MODEL`** (text; default `Qdrant/BM25`) — Local FastEmbed sparse model for the keyword (BM25) leg of hybrid search.
- **`DENSE_QUERY_INSTRUCTION`** (text; default `Represent this sentence for searching relevant passages: `) — Prepended to the dense query text only (never the corpus side) before embedding. bge-small's model card recommends this instruction for short-query→passage retrieval, and fastembed does not apply it for us. Empty string disables it. Query-side only, so changing it needs no re-ingest.
- **`DENSE_PASSAGE_PREFIX`** (text; default empty) — Prepended to every \*corpus\* text before dense embedding (never to queries, never to BM25). Empty for bge; "passage: " for the e5 family, which is trained on that pair with DENSE_QUERY_INSTRUCTION="query: ". Recorded with the collection: changing it is a new vector space, like changing the model.
- **`SPARSE_LANGUAGE`** (text; default `english`) — BM25's stemmer language (Qdrant/BM25 only), e.g. "german", "french", "spanish". Recorded with the collection like the models: a change needs a re-index.
- **`EMBEDDING_MODEL_TAG`** (text; default `fastembed:BAAI/bge-small-en-v1.5`) — Stored with the collection; a mismatch with this value refuses questions until the corpus is re-indexed for the current embedding setup.
- **`RERANKER_MODEL`** (text; default `jinaai/jina-reranker-v2-base-multilingual`) — Local cross-encoder that rescores the fused candidates.
- **`RERANKER_ONNX_FILE`** (text; default `auto`) — Which ONNX export of RERANKER_MODEL to load. "auto" (default): jina-v2's int8 export, fastembed's registered file for any other model. "" : always fastembed's registered file (jina: fp32 onnx/model.onnx). Anything else: that path in the repo. int8 jina measured 2.5x faster than fp32 with 40% less memory, 81-83% of the same top-6 context and no question losing its context (README "Latency").
- **`RERANKER_BATCH_SIZE`** (integer; default `8`; > 0) — Caps how many query-document pairs the cross-encoder scores per forward pass, which is what bounds its peak memory: attention cost grows with batch x sequence^2, and at CHUNK_SIZE_TOKENS=448 the sequences are long. Peak memory is a function of THIS VALUE ALONE, not of its ratio to anything else. Measured over 50 candidates x ~448 tokens: batch 8 peaked at 2.66 GB (exactly the loaded-model baseline — scoring adds nothing on top), batch 16 at 3.53 GB, batch 50 at 6.33 GB, which OOM-killed (exit 137) the container on a default ~6 GB Docker Desktop the first time anyone asked a whole-corpus question. Latency was flat across the whole range (~45-49s), so a larger batch buys no speed and no answer quality — it only changes how the same work is chunked, and what it peaks at while doing so. Raise it only against known memory headroom; api/main.py logs a warning at startup past the measured-safe point. Keeping it &lt;= rerank_candidates is a SEPARATE and weaker rule: above that, batching is simply a no-op because there aren't enough pairs to fill a batch. That relation is not the memory guard, and satisfying it is not sufficient — batch 50 against 50 candidates satisfies it and still peaks at 6.33 GB. The historical incident (a default of 64 against 50 candidates) happened to violate both rules at once, which is what made the relation look like the cause.
- **`EMBEDDING_THREADS`** (integer; default `0`; ≥ 0) — ONNX Runtime intra-op threads per session; 0 keeps ORT's default (all cores). Lowering these makes a lone question SLOWER — their value is stopping ingest embedding and query-time rerank from oversubscribing the same cores when they overlap. Leave at 0 unless you run ingest and questions concurrently.
- **`RERANKER_THREADS`** (integer; default `0`; ≥ 0) — ONNX Runtime threads for the reranker's session; see EMBEDDING_THREADS.
- **`RERANK_CACHE_SIZE`** (integer; default `4096`; ≥ 0) — Rerank scores memoized per (query, passage text). Regenerate/Retry re-ask the identical query, so a hit skips the most expensive stage entirely. 0 disables.
- **`WARMUP_MODELS`** (true/false; default false) — Load the embedding and reranking models at startup instead of on the first request. Off by default: the embedding/reranker models are lazy-loaded on first use, so a cold-start warmup at boot moves that (potentially tens-of-seconds) cost earlier, surfacing a misconfigured MODEL name at startup instead of on a user's first question. Left opt-in rather than on-by-default because the warmup path reads this setting directly (not through FastAPI's DI), which sidesteps test fixtures' dependency overrides — an operator-set `true` in a real `.env` would otherwise also trigger real model downloads during local test runs against that same file. Note: the default reranker (jina-reranker-v2, int8 export) is ~0.28 GB (~1.1 GB for fp32) — enabling warmup or asking the first question after a config change pays that download cost once.

## Retrieval and ranking

Hybrid search, fusion, reranking and which passages reach the prompt.

- **`RRF_K`** (integer; default `60`; > 0) — Reciprocal Rank Fusion constant for combining the dense and sparse lists (fixed at 60 by design; never overridden per request).
- **`DENSE_RETRIEVAL_LIMIT`** (integer; default `50`; > 0) — Candidates fetched from the dense (semantic) leg.
- **`SPARSE_RETRIEVAL_LIMIT`** (integer; default `50`; > 0) — Candidates fetched from the sparse (BM25) leg.
- **`FUSED_TOP_N`** (integer; default `50`; > 0) — Candidates kept after fusion, before reranking.
- **`RERANK_CANDIDATES`** (integer; default `50`; > 0) — How many of the fused top-N candidates are actually sent through the cross-encoder — the reranker is the single most expensive retrieval-side stage, so this lets an operator trade candidate coverage for latency without touching fused_top_n (which stays "the fusion spec's own top-N" per CLAUDE.md). Clamped to fused_top_n at the call site — it can never rerank more than fusion produced.
- **`RERANK_TOP_K`** (integer; default `8`; > 0) — Passages kept after reranking (before the context cap below).
- **`MAX_CONTEXT_CHUNKS`** (integer; default `6`; > 0; *per workspace*) — Most passages put into the answering prompt.
- **`RERANK_MIN_SCORE`** (number; default `0.15`; ≥ 0.0, ≤ 1.0; *per workspace*) — Lowest rerank score (0 to 1) a passage needs to be used for an answer. Cross-encoder logits are unbounded and model-specific; sigmoid-normalizing to [0, 1] before comparing against this threshold makes it comparable across reranker models. sigmoid(0)=0.5 is the "indifferent" point, but in practice jina-reranker-v2's scores for genuinely relevant pairs run well below that (a clean direct-hit query scored ~0.6, a paraphrase-only match ~0.2) — 0.30 was too strict and returned "insufficient context" even when loosely relevant content existed. 0.15 still blocks clearly off-topic chunks (scored &lt;0.05 in the same real-world trace) while letting paraphrase-level matches through.
- **`RERANK_FALLBACK_COUNT`** (integer; default `3`; ≥ 0, ≤ 20; *per workspace*) — When no candidate clears rerank_min_score, this many of the best-ranked ones go to the model anyway, marked as weak matches, and it decides whether they answer the question (the answer then carries weak_matches). 0: refuse at once, as before. Each cross-encoder scores on its own scale, so a fixed floor dropped the passage that answered the question: "what are foundational models" scored 0.006 with MiniLM and 0.154 with jina against "Foundation models: You have trained ...".
- **`RERANK_RELATIVE_FLOOR`** (number; default `0.0`; ≥ 0.0, ≤ 1.0; *per workspace*) — Opt-in (0 = off): of the passages that cleared the floor, keep only those scoring within this share of the way from the median candidate's score to the best one ((score - median) / (top - median) &gt;= this), and always the best two. Works on any reranker's scale, and only ever trims the prompt — never turns an answer into a refusal. A low-scoring sixth passage is ~1k tokens of prompt to read.
- **`RECENCY_BOOST_WEIGHT`** (number; default `0.0`; ≥ 0.0, ≤ 1.0; *per workspace*) — Favour newer documents among the passages kept for an answer: each is ordered by its rerank score plus this times how recent its document is (1 for the newest kept, 0 for one a year or more older). Nothing is added or dropped, and RRF is untouched; the order decides which passages reach the prompt first. 0 = off.
- **`CONTEXT_NEIGHBOR_RADIUS`** (integer; default `1`; ≥ 0) — After rerank, each selected chunk's context is expanded with up to this many neighboring chunks (by chunk_ordinal) on each side from the same document — small-to-big retrieval: rerank on tight chunks, generate on richer context. 0 disables expansion entirely.
- **`CONTEXT_NEIGHBOR_MAX_CHARS`** (integer; default `0`; ≥ 0) — Caps the neighbour text added on each side of a chunk, in characters: the text nearest the chunk is kept, cut at a word boundary. 0 = no cap (the default). Neighbours are most of the prompt, and prefill is most of the wait before the first token. 300 was measured only at CHUNK_SIZE_TOKENS=96 (about 1s less prefill on qwen2.5, same answers on 11 questions); at the default chunk size it would drop most of each neighbour, which hasn't been measured (README "Latency").
- **`CONTEXT_DIVERSITY_ENABLED`** (true/false; default true) — Post-rerank diversity/dedup: after scoring, drop a survivor that is a near- duplicate of a higher-ranked kept chunk (same-document neighbor within context_neighbor_radius ordinals, whose expanded contexts would nearly coincide, OR word-shingle Jaccard &gt;= context_diversity_max_similarity). Freed slots refill from lower-ranked survivors, so context still gets up to rerank_top_k \*distinct\* chunks. Pure selection after scoring — rerank scores/order are untouched.
- **`CONTEXT_DIVERSITY_MAX_SIMILARITY`** (number; default `0.6`; > 0.0, ≤ 1.0) — Word-shingle similarity at which a passage counts as a near-duplicate.
- **`QUERY_EXPANSION_ENABLED`** (true/false; default false) — Multi-query expansion (default OFF): when enabled, one LLM call rewrites the query into query_expansion_count alternative phrasings; each is embedded and hybrid-searched, and the result lists are RRF-fused (same pinned rrf_k) before a single rerank. Adds one LLM round-trip per question, so it stays opt-in.
- **`QUERY_DECOMPOSITION_ENABLED`** (true/false; default false) — Compound-question decomposition (default OFF): a question that asks about several things ("the leave, expense and IT policies") is split by one LLM call into standalone parts, each searched and reranked on its own, their passages taken in turn — scored against the whole question, the cross-encoder ranked each part's passages under the bar. Only questions that look compound (and, a comma, versus, compare, both, between) make the call.
- **`QUERY_DECOMPOSITION_MAX_PARTS`** (integer; default `3`; ≥ 2, ≤ 5) — Most parts a question is split into.
- **`QUERY_DECOMPOSITION_MAX_TOKENS`** (integer; default `160`; > 0) — Output budget of the decomposition call.
- **`QUERY_EXPANSION_COUNT`** (integer; default `3`; ≥ 1, ≤ 8) — Alternative phrasings asked for when QUERY_EXPANSION_ENABLED.
- **`QUERY_EXPANSION_MAX_TOKENS`** (integer; default `192`; > 0) — Output budget of the query-expansion call.

## Generation providers

The LLM that writes answers. Hosted providers are optional add-ons.

- **`LLM_PROVIDER`** ("ollama" or "openai" or "openai_compatible" or "anthropic"; default `ollama`) — Which provider writes answers: ollama (local, default), openai, openai_compatible or anthropic. Checked at startup (case-insensitive): a typo used to boot fine and then fail every question with a 400 that blamed the client.
- **`ALLOWED_REQUEST_PROVIDERS`** (list of "ollama" or "openai" or "openai_compatible" or "anthropic" or unset; default unset) — Providers a question may pick per request (`llm_provider` on /questions\*), as a JSON list, e.g. ["ollama"]. Unset: any configured provider. LLM_PROVIDER itself is always allowed. Restricting it stops a caller — anyone with the read key, or anyone at all without API_KEY — from switching to a paid provider at will.
- **`LLM_MODEL`** (text; default `llama3.1:8b`) — Generation model for LLM_PROVIDER=ollama (an Ollama model name).
- **`LLM_TEMPERATURE`** (number; default `0.0`; ≥ 0.0, ≤ 2.0) — Sampling temperature for answers (0 = most deterministic).
- **`LLM_MAX_TOKENS`** (integer; default `512`; > 0) — Output-token budget of an answer (reasoning models get REASONING_TOKEN_HEADROOM on top).
- **`LLM_REQUEST_TIMEOUT_SECONDS`** (number; default `60.0`; > 0.0) — Longest wait for an LLM reply; local providers add prompt-reading time (below).
- **`LLM_PREFILL_TOKENS_PER_SECOND`** (number; default `40.0`; ≥ 0.0) — Local providers (Ollama, OpenAI-compatible) send nothing while they read the prompt, and LiteLLM hands its timeout to httpx as the longest silence allowed (it turns an httpx.Timeout back into one number for Ollama), so a 6k-token prompt on an M1 timed out before the first token. Each call's timeout grows by its estimated prompt tokens divided by this rate (or by the slower rate this server has been measured at). 0 turns the scaling off; hosted providers never scale.
- **`LLM_NUM_RETRIES`** (integer; default `0`; ≥ 0) — LiteLLM retries of a failed LLM call (0 = none).
- **`LLM_CONTEXT_WINDOW`** (integer; default `0`; ≥ 0) — The model's context window in tokens, for fitting the prompt into it. 0 (default) uses OLLAMA_NUM_CTX when the provider is Ollama and that is set; for OpenAI and Anthropic models, the window LiteLLM's bundled model table gives; else no fitting. When known, the prompt drops the lowest-ranked context and then the oldest history until it fits with LLM_MAX_TOKENS to spare — an over-long prompt had its \*front\* (the grounding rules) silently cut by the model server.
- **`OLLAMA_BASE_URL`** (text; default `http://localhost:11434`) — Where Ollama listens (Compose's ollama profile: http://ollama:11434).
- **`OLLAMA_KEEP_ALIVE`** (text; default `30m`) — Ollama unloads a model from memory 5 minutes after its last use by default, forcing a multi-second reload on the next question. Keeping it resident trades idle RAM for eliminating that reload on every request after the first.
- **`OLLAMA_NUM_CTX`** (integer; default `0`; ≥ 0) — Context window requested from Ollama; 0 leaves it to Ollama. Not a speed knob — measured on an M1, 8192 vs Ollama's 32768 left prefill/decode speed unchanged but cut the loaded model from 8.2 GB to 5.3 GB. Too small silently truncates the front of the prompt (the grounding rules), so leave headroom for history.
- **`OLLAMA_MAX_PARALLEL`** (integer; default `1`; ≥ 0) — How many LLM calls this process sends Ollama at once. Ollama answers one request at a time unless OLLAMA_NUM_PARALLEL says otherwise and queues the rest — where each waiting request's timeout runs, so a second question timed out without ever being read. Calls past this wait here instead (the stream says so), and their timeout starts when they are sent. 0 = no limit (match a raised OLLAMA_NUM_PARALLEL).
- **`OPENAI_API_KEY`** (text or unset; default unset) — OpenAI key, for LLM_PROVIDER=openai or a question that picks it. Hosted and optional: uploads and indexing never call it.
- **`OPENAI_MODEL`** (text; default `gpt-4o-mini`) — Generation model for LLM_PROVIDER=openai.
- **`OPENAI_REASONING_EFFORT`** (text; default `low`) — Reasoning models (gpt-5 family) spend hidden reasoning tokens out of the same max_completion_tokens budget as the visible answer. "low" keeps that spend modest while still reasoning; "minimal" nearly eliminates it. "" leaves the provider default. Only applied to models LiteLLM reports as reasoning-capable.
- **`REASONING_TOKEN_HEADROOM`** (integer; default `3072`; > 0) — Extra output-token budget added ON TOP of llm_max_tokens for reasoning models, so hidden reasoning tokens don't starve the visible answer (a 512 budget is entirely consumed by reasoning on non-trivial questions, yielding an empty answer). Ignored for non-reasoning models, which keep the plain llm_max_tokens budget.
- **`OPENAI_COMPATIBLE_BASE_URL`** (text; default empty) — Any server speaking the OpenAI chat-completions API — vLLM, llama.cpp's server, LM Studio, LocalAI — selected with LLM_PROVIDER=openai_compatible (or per request). Self-hostable like Ollama; the base URL is the one ending in /v1. The key is optional: most local servers accept anything.
- **`OPENAI_COMPATIBLE_MODEL`** (text; default empty) — Model name sent to the OpenAI-compatible server.
- **`OPENAI_COMPATIBLE_API_KEY`** (text; default empty) — Key for the OpenAI-compatible server, if it wants one.
- **`ANTHROPIC_API_KEY`** (text or unset; default unset) — Optional Anthropic (Claude) generation, selected with LLM_PROVIDER=anthropic or per question. Like OpenAI it is a hosted add-on: upload, indexing and passage search never call it, and embeddings stay local.
- **`ANTHROPIC_MODEL`** (text; default `claude-sonnet-5-5`) — Generation model for LLM_PROVIDER=anthropic.

## Answers and conversations

Prompt size, conversation memory, citations, follow-ups and caching.

- **`PROMPT_TOKEN_BUDGET`** (integer; default `0`; ≥ 0) — Opt-in (0 = off): the most estimated tokens a question's prompt may take, whatever the window. Over it, neighbour text goes first (lowest-ranked passage first), then whole lowest-ranked passages, then the oldest history. A local model reads a prompt at ~70 tokens/s on an M1, so prompt size is most of the wait.
- **`ANSWER_LANGUAGE`** (text; default empty; at most 40 characters; *per workspace*) — The language answers are written in. Empty: the model's choice (usually the question's, often English with English instructions); "match": the question's language; or a language name ("German"). One fixed sentence in the system prompt, so prompt prefix caching still applies.
- **`ANSWER_FORMAT`** ("default" or "bullets" or "table" or "brief"; default `default`; *per workspace*) — How answers are laid out: "default" (the model's choice), "bullets", "table" (a Markdown table where the content allows it) or "brief" (two or three sentences). One fixed sentence after the grounding rules, which it never replaces. A question can ask for another (`answer_format`).
- **`CONVERSATION_CONDENSE_ENABLED`** (true/false; default true) — Conversation memory: the client sends recent prior turns with each question (the server stays stateless — no session store). When history is present, one extra LLM call rewrites the follow-up into a standalone retrieval query before embedding/search/rerank, so "what about the second one?" doesn't get embedded literally. Generation always sees the raw history either way; this only gates the condense step.
- **`CONVERSATION_MAX_HISTORY_MESSAGES`** (integer; default `12`; ≥ 0) — Server-side truncation cap (most recent messages kept) — independent of the request schema's own cap, so an operator can tighten it without a schema change.
- **`GENERATION_HISTORY_ASSISTANT_CHARS`** (integer; default `0`; ≥ 0) — Opt-in (0 = off): earlier answers sent to the answering model are cut to this many characters each (the condense step keeps its own 1500). The question already carries its resolved meaning ("Read with the conversation as: ...").
- **`CONDENSE_MAX_TOKENS`** (integer; default `128`; > 0) — The condense call only needs to produce one rewritten sentence — capped well below llm_max_tokens so a runaway rewrite can't burn the generation token budget.
- **`CONDENSE_CACHE_SIZE`** (integer; default `256`; ≥ 0, ≤ 100000) — Follow-up rewrites remembered (the same question, history, provider and model): Retry, an edited follow-up asked again and other tabs skip the condense call. 0 = off.
- **`CITATION_RETRY_ENABLED`** (true/false; default true) — Zero-citation retry (default ON): if a non-insufficient answer comes back with no [n] citation markers though sources were available, retry once with a stricter reminder to cite. Never loops; falls back to the original answer if the retry still lacks citations. See api/generation.py.
- **`CITATION_RETRY_MODE`** ("auto" or "llm" or "backfill" or "off"; default `auto`) — How an uncited answer gets its citations. "llm": the retry above (a second full generation — about a minute on a local model). "backfill": each answer sentence a source plainly states (most of its content words, in one or two adjacent sentences of the source) gets that source's [n]; no LLM call. "auto" (default): backfill first for local providers (Ollama, OpenAI-compatible), the LLM retry only when nothing matched; the LLM retry for hosted providers. "off": neither (as CITATION_RETRY_ENABLED=false).
- **`AUTO_CONTINUE_MAX`** (integer; default `0`; ≥ 0, ≤ 5) — A streamed answer cut off at the output limit is continued in the same stream, up to this many times (0 = off: the reader presses Continue).
- **`ANSWER_SELF_CHECK`** (true/false; default true) — Each answer's sentences checked against the passages it cites (no LLM, no search): `verification` on /questions and the stream's `done` lists what a passage supports, what none does, and what states other numbers, for the UI to mark.
- **`FOLLOWUPS_ENABLED`** (true/false; default false; *per workspace*) — Suggested follow-up questions (default OFF: one more LLM call per answer, which on a local model is seconds). When on, an answer that cited its sources gets up to FOLLOWUP_COUNT short questions those sources can answer, sent after the answer (a `followups` frame after `done` on the stream; `followups` on /questions).
- **`FOLLOWUP_COUNT`** (integer; default `3`; ≥ 1, ≤ 5) — Most follow-up questions suggested per answer.
- **`FOLLOWUP_MAX_TOKENS`** (integer; default `128`; > 0) — Output budget of the follow-up call.
- **`ANSWER_CACHE_SIZE`** (integer; default `256`; ≥ 0) — Finished answers kept for repeat questions (api/answer_cache.py); 0 turns the cache off. Invalidated by every ingest, delete and tag change, and expired after the TTL. Questions with conversation history are never cached.
- **`ANSWER_CACHE_TTL_SECONDS`** (number; default `3600.0`; > 0.0) — How long a cached answer (and a cached LLM reply) is reused.
- **`PROMPT_CACHE_SIZE`** (integer; default `256`; ≥ 0, ≤ 100000) — Replies to identical LLM calls (the same messages, model and options) reused for ANSWER_CACHE_TTL_SECONDS, whatever the corpus did meanwhile: retrieval, citations and the trace still run; only the call is saved. 0 = off.

## Summaries

Whole-document summaries, comparisons and briefs.

- **`SUMMARY_BATCH_CHARS`** (integer; default `9000`; ≥ 1000, ≤ 200000) — Whole-document summaries and comparisons (POST /questions/summary): the document's passages are read in order, in parts of at most SUMMARY_BATCH_CHARS characters (lowered to fit LLM_CONTEXT_WINDOW / OLLAMA_NUM_CTX when known), each part is turned into cited notes, and the notes are combined into one answer. At most SUMMARY_MAX_BATCHES parts per request: a longer document is read at evenly spaced passages, and the answer reports how many it read. Each part is one LLM call, so this bounds the time a summary takes. SUMMARY_MAX_TOKENS is the final answer's budget (summaries run longer than answers); notes keep LLM_MAX_TOKENS.
- **`SUMMARY_MAX_BATCHES`** (integer; default `8`; ≥ 1, ≤ 64) — Most parts read per summary request (see above).
- **`SUMMARY_MAX_TOKENS`** (integer; default `1024`; > 0) — Output budget of a summary's final answer (see above).
- **`SUMMARY_PARALLEL_PARTS`** (integer; default `4`; ≥ 1, ≤ 16) — Parts of a long document turned into notes at once, with a hosted provider (OpenAI, Anthropic, an OpenAI-compatible server). Ollama always takes them one at a time: it runs one generation at a time anyway, and parallel requests only queue there.
- **`SUMMARY_MAX_DOCUMENTS`** (integer; default `5`; ≥ 2, ≤ 10) — Most documents one brief (POST /questions/summary, "mode": "brief") may cover.

## Documents and chunking

Upload limits, parsing limits and how text is cut into passages.

- **`MAX_UPLOAD_BYTES`** (integer; default `52428800`; > 0) — Largest single upload (and URL download) accepted.
- **`MAX_IMPORT_BYTES`** (integer; default `268435456`; > 0) — POST /import: the largest backup zip (GET /export) accepted. Each original inside is still held to max_upload_bytes.
- **`MAX_MIGRATE_BYTES`** (integer; default `8589934592`; > 0) — Largest migration bundle POST /admin/migrate/import accepts (every workspace's originals and index snapshot in one file).
- **`MAX_MIGRATE_MEMBERS`** (integer; default `1000000`; > 0) — Entries each backup inside a migration bundle may hold (one per original and per kept version): an uploaded backup's cap (200,000) refused a large corpus's bundle. Reading a zip's directory takes about 500 bytes per entry.
- **`MAX_ARCHIVE_MEMBERS`** (integer; default `500`; > 0) — POST /documents/archive: a zip of documents, each indexed as its own. At most this many members are looked at; what they expand to in all is held to MAX_IMPORT_BYTES, and each one to MAX_UPLOAD_BYTES. The zip itself may be up to MAX_IMPORT_BYTES too (request_guard.body_limit).
- **`MIN_FREE_BYTES`** (integer; default `0`; ≥ 0) — Refuse uploads, imports and backups (507) while the disk they write to has less than this free; 0 = no check.
- **`MAX_CHUNKS_PER_DOCUMENT`** (integer; default `0`; ≥ 0) — Fail a document that chunks into more passages than this (0 = no limit): one enormous file otherwise holds an indexing worker for hours.
- **`FRONT_MATTER_TAGS`** (true/false; default false) — A Markdown file's front matter ``tags:`` become its document's tags when it is indexed (new or changed) without tags of its own.
- **`INDEX_HIDDEN_CONTENT`** (true/false; default false) — Index what an Office file hides from its reader: hidden and very hidden XLSX sheets, DOCX text formatted hidden, hidden PPTX slides. Off: they are left out, as the HTML parser leaves out hidden elements (a hidden sheet of salaries became citable text). Applies when a document is next indexed.
- **`AUTO_REBUILD_LOST_INDEX`** (true/false; default false) — At startup, a workspace whose index holds no document while its originals are kept (a lost Qdrant volume) is reported (NOTIFY_URL, event index_lost); with this on it is also rebuilt from those originals, as POST /admin/rebuild would.
- **`CHUNK_SIZE_TOKENS`** (integer; default `448`; > 0) — Largest passage, in the dense model's tokens. Real dense-model subword tokens (api/chunking.py counts with the model's own tokenizer). bge-small hard-truncates at 512 tokens; the chunker reserves headroom for the contextual-embedding prefix and never exceeds this budget, so the embedded string fits the model. 448 leaves margin under 512 for special tokens and the prefix. Changing this alters chunk contents — re-upload documents for consistency.
- **`CHUNK_OVERLAP_TOKENS`** (integer; default `100`; ≥ 0) — Tokens repeated between consecutive chunks of a section (must be below CHUNK_SIZE_TOKENS; clamped to half the window).
- **`MIN_SECTION_WORDS`** (integer; default `40`; ≥ 0) — Markdown heading sections shorter than this are folded into a neighboring section before chunking, so a one-line subsection doesn't become its own low-information, hard-to-retrieve chunk.
- **`CSV_ROWS_PER_SECTION`** (integer; default `50`; ≥ 1) — CSV rows per section: a CSV upload is grouped into sections of this many data rows (each row rendered as "col: value; ..."), so citations point at a labeled row range instead of one giant section.
- **`MAX_PDF_PAGES`** (integer; default `2000`; ≥ 1) — PDF work limits. A PDF over max_pdf_pages is refused (400) before any page is read; only the first max_ocr_pages text-less pages are OCR'd (200 dpi each, the slowest step by far), the rest skipped with a warning. Without them a small file of thousands of blank pages held both ingest workers for hours.
- **`MAX_OCR_PAGES`** (integer; default `100`; ≥ 0) — Most text-less PDF pages OCR'd per document (see above).
- **`MAX_SPREADSHEET_CELLS`** (integer; default `200000`; ≥ 1) — An XLSX workbook with more cells than this (counted across sheets, empty ones included as openpyxl reads them) is refused. Every cell is held as text while the sheet is rendered; this bounds a small file that declares a huge sheet.
- **`PARSE_TIMEOUT_SECONDS`** (number; default `300.0`; ≥ 0) — Each upload is parsed in a child process that is stopped after this many seconds (api/parse_isolation.py): a pathological file used to hold an ingest worker indefinitely, competing with questions for the GIL. 0 parses in the API process with no limit.
- **`PARSE_MEMORY_LIMIT_MB`** (integer; default `0`; ≥ 0) — An address-space cap (MB) on that child process, Linux only (ignored elsewhere and when PARSE_TIMEOUT_SECONDS is 0): a parser running away fails the one document with a named error instead of drawing the container's OOM killer. 0 = no cap. Address space is larger than resident memory (native libraries map a lot); leave headroom.

## Indexing jobs

The ingest queue, job records and background re-indexing.

- **`INGEST_BATCH_SIZE`** (integer; default `64`; > 0) — Chunks are embedded and upserted in batches of this size rather than all at once, so a single huge document reports incremental job progress and never holds one giant embedding call in memory.
- **`INGEST_WORKERS`** (integer; default `2`; ≥ 1, ≤ 32) — Documents indexed at the same time (the ingest pool's threads). Each one holds a document's parsed text and vectors in memory, and they share the embedding model: more is faster on a big folder only with cores and memory to spare.
- **`INGEST_UPSERT_BATCH_SIZE`** (integer; default `256`; > 0) — Points written per Qdrant upsert, independent of the embedding batch above. Each upsert is a round trip with wait=True under the filename lock, so writing a 1,000-chunk document in 64-point calls held the lock through 16 of them.
- **`INGEST_REUSE_VECTORS`** (true/false; default true) — A re-upload, re-index, URL refresh or watched-folder change reuses the stored vectors of every chunk that didn't change (same point id — filename, page, section and text — and same chunking fingerprint) and embeds only the rest. False embeds every chunk again.
- **`INGEST_MAX_PENDING_BYTES`** (integer; default `536870912`; > 0) — Upload bytes allowed to sit queued or in progress at once. Every accepted upload holds its full content in memory until its ingest job finishes, and the executor queue itself is unbounded — past this budget uploads get 503 + Retry-After instead of growing memory without limit. An empty backlog always admits one.
- **`INGEST_JOBS_MAX_RETAINED`** (integer; default `50`; > 0) — Finished background ingest jobs beyond this count are pruned oldest-first, so a long-running process doesn't accumulate unbounded job history in memory — same tradeoff/pattern as trace_max_retained below. It also caps the jobs queued or running at once (IngestBacklog): past it an upload gets 503 + Retry-After even with byte budget left, so the store never evicts a job still in flight.
- **`JOB_STORE_BACKEND`** ("memory" or "sqlite"; default `memory`) — Where ingest job status is kept: "memory" or "sqlite" (JOB_STORE_PATH). "memory" (default) matches every prior release: jobs are lost on restart, and a client polling a job whose process just restarted gets a 404 even if the ingest actually succeeded in Qdrant. "sqlite" persists job status to job_store_path (stdlib sqlite3, no new dependency) so that poll survives a restart. Traces remain in-memory-only either way — see api/tracing.py's TraceStore docstring.
- **`JOB_STORE_PATH`** (text; default `./data/jobs.db`) — Only read when job_store_backend="sqlite". The containing directory (created if missing) must be writable (in Docker, under /app/data: the docrag_data volume); startup fails naming the setting otherwise (api/storage.py).
- **`REQUEUE_INTERRUPTED_JOBS`** (true/false; default true) — With JOB_STORE_BACKEND=sqlite: a re-index or URL fetch a restart cut off is started again (once) at startup, as POST /documents/jobs/{id}/retry would.
- **`AUTO_REINDEX_STALE`** (true/false; default false) — Re-index stale documents in the background (api/auto_reindex.py): after an upgrade that changes chunking, or a chunk-setting change, every document with a stored original (RAW_DOCUMENT_DIR) is brought up to date without anyone clicking "re-index all". At most AUTO_REINDEX_BATCH of its jobs run at once, so uploads still get through. Off by default.
- **`AUTO_REINDEX_BATCH`** (integer; default `4`; > 0) — Most of the background re-index's jobs running at once.
- **`IMPORT_SPOOL_DIR`** (text; default empty) — Where background imports keep their uploaded zip while they run; with it set, an import a restart cut off runs again at startup. Empty: the system temp directory, and a restart forgets them.
- **`DOCUMENT_LISTING_TTL_SECONDS`** (number; default `300.0`; > 0) — The document listing (GET /documents) is read by scrolling every point, and kept between changes made by this process, which it updates in place. This only bounds how long a change made by \*another\* process (a second API worker, the eval harness's ingest) can go unseen. It was 30 s, which made the panel rescan a large collection twice a minute for writes a single-process deployment never has.

## Originals, backups and workspaces

Keeping uploaded files, scheduled backups and separate corpora.

- **`RAW_DOCUMENT_DIR`** (text or unset; default unset) — Keep each uploaded original in this directory (needed for re-index, backups with originals, versions and rebuilds). Unset = originals are discarded. None (default) matches every prior release: an uploaded document's raw bytes are discarded once ingestion is queued — no "download original", and a future chunking/embedding change needs every document manually re-uploaded. Setting this to a directory keeps a copy of each upload (keyed by its normalized basename filename) and exposes it via GET /documents/{filename}/original. Same writable-directory/volume requirement as job_store_path. It also enables POST /documents/{filename}/reindex (and the bulk POST /documents/reindex), which re-chunk and re-embed a document from its stored original.
- **`RAW_DOCUMENT_VERSIONS`** (integer; default `0`; ≥ 0, ≤ 50) — With RAW_DOCUMENT_DIR: how many earlier versions of each original a re-upload keeps (GET /documents/{f}/versions, POST .../versions/{v}/restore). 0 = none (default).
- **`BACKUP_DIR`** (text; default empty) — Scheduled backups (api/scheduled_backup.py): the /export zip of every workspace written to BACKUP_DIR/&lt;workspace&gt;/ every BACKUP_INTERVAL_HOURS, the newest BACKUP_KEEP of each kept. Empty = off.
- **`BACKUP_INTERVAL_HOURS`** (number; default `24`; > 0) — Hours between scheduled backups of each workspace.
- **`BACKUP_KEEP`** (integer; default `7`; > 0) — Scheduled backups kept per workspace (newest first).
- **`BACKUP_COPY_COMMAND`** (text; default empty) — Run after each scheduled backup is written and checked, with the backup's path as its last argument (no shell: e.g. "rclone copy --no-traverse" plus a remote, or "cp" plus a mounted share's folder) — a copy somewhere other than this disk. Empty = none. A command that fails or runs past BACKUP_COPY_TIMEOUT_SECONDS fails the backup (reported like any other failure).
- **`BACKUP_COPY_TIMEOUT_SECONDS`** (integer; default `1800`; > 0) — How long BACKUP_COPY_COMMAND may run before the copy (and the backup) fails.
- **`WORKSPACES_ENABLED`** (true/false; default false) — Separate corpora on one server (api/workspaces.py), off by default: each is its own Qdrant collection (QDRANT_COLLECTION__ws_&lt;name&gt;) and originals directory, chosen per request with the X-Workspace header and created with POST /workspaces. Not access control — any key that can use the API can use every workspace.
- **`WORKSPACE_SETTINGS_PATH`** (text; default empty) — Per-workspace relevance settings (PUT /workspaces/{name}/settings) are kept here (sqlite) to survive a restart; empty keeps them in memory only.

## Watched folders

Keeping the index in step with folders on the server.

- **`WATCH_DIR`** (text; default empty) — Keep a workspace's index in step with a folder (api/watch.py): every WATCH_INTERVAL_SECONDS, new and changed supported files are indexed like uploads. Empty (default) = off. WATCH_DELETE_MISSING also deletes the document of a file that disappears — only ones the watcher saw in the folder itself. WATCH_WORKSPACE names the workspace it indexes into (needs WORKSPACES_ENABLED; created at startup if it doesn't exist yet); empty = the default workspace.
- **`WATCH_DIRS`** (text; default empty; at most 8000 characters) — More watched folders, each into its own workspace: "path=&gt;workspace;path2=&gt;ws2" (no "=&gt;workspace": the default one). Alongside WATCH_DIR, with the same interval, filters and deletion rule; the same folder twice, or one inside another, stops startup.
- **`WATCH_INTERVAL_SECONDS`** (number; default `30.0`; ≥ 2.0, ≤ 86400.0) — Seconds between scans of the watched folders.
- **`WATCH_DELETE_MISSING`** (true/false; default false) — Delete the document of a watched file that disappears (only files the watcher saw in the folder itself).
- **`WATCH_KEEP_PATHS`** (true/false; default false) — Name a watched file by its path under the folder ("reports/2024/q1.pdf") instead of its base name, so same-named files in different folders are all indexed.
- **`WATCH_WORKSPACE`** (text; default empty) — Workspace WATCH_DIR indexes into (needs WORKSPACES_ENABLED); empty = default.
- **`WATCH_MAX_RETRIES`** (integer; default `3`; ≥ 0, ≤ 100) — A watched file whose read fails, or whose indexing job fails, is tried again after WATCH_INTERVAL_SECONDS x 2^n (at most WATCH_RETRY_MAX_SECONDS), up to WATCH_MAX_RETRIES times; then it waits until the file changes. 0 = never retried.
- **`WATCH_RETRY_MAX_SECONDS`** (number; default `3600.0`; ≥ 2.0, ≤ 604800.0) — Longest wait between retries of a failing watched file (see above).
- **`WATCH_INCLUDE`** (text; default empty; at most 4000 characters) — Which files under WATCH_DIR are watched: comma-separated glob patterns. One with no "/" matches a file name ("\*.py"), one with a "/" the path under WATCH_DIR ("drafts/\*"). With WATCH_INCLUDE set only matching files are; WATCH_EXCLUDE leaves matching files (and directories, by name) out. An excluded file is never deleted by WATCH_DELETE_MISSING. Empty = every supported file.
- **`WATCH_EXCLUDE`** (text; default empty; at most 4000 characters) — Glob patterns of watched files and directories to leave out (see above).

## Web sources

Documents fetched from URLs, sitemaps and feeds.

- **`URL_INGEST_ENABLED`** (true/false; default false) — POST /documents/from-url (off by default): the server downloads a document from a URL and indexes it like an upload (MAX_UPLOAD_BYTES applies). Only public addresses are fetched unless URL_INGEST_ALLOW_PRIVATE is set — the request comes from inside the deployment, so a private one could reach Qdrant or cloud metadata.
- **`URL_INGEST_ALLOW_PRIVATE`** (true/false; default false) — Allow URL ingest from private addresses (see above).
- **`URL_INGEST_TIMEOUT_SECONDS`** (number; default `30.0`; > 0.0, ≤ 600.0) — Time allowed for one URL download, start to finish.
- **`URL_REFRESH_INTERVAL_MINUTES`** (integer; default `0`; ≥ 0, ≤ 525600) — Refresh URL-fetched documents in the background (api/url_refresh.py): each one whose last check is this many minutes old is fetched again conditionally (an unchanged page is a 304, nothing re-indexed), at most URL_REFRESH_BATCH per pass. 0 = off. Needs URL_INGEST_ENABLED.
- **`URL_REFRESH_BATCH`** (integer; default `4`; > 0) — Most refreshed documents queued per pass.
- **`URL_REFRESH_STATE_PATH`** (text; default empty) — Where the scheduled refresh keeps when each document was last checked (a sqlite file), so a restart doesn't make every document due at once. Empty = in memory.
- **`SITEMAP_MAX_PAGES`** (integer; default `100`; > 0) — POST /documents/from-sitemap: the most pages one sitemap or feed adds.

## Access and limits

Keys, CORS and how much work callers may start.

- **`API_KEY`** (text; default empty) — Optional single shared API key. Empty (the default) keeps every route fully open, preserving the zero-config self-host story. When set, every route except /health, /health/ready and /config — reads included — requires a matching X-API-Key header (api/dependencies.py's require_api_key, and api/request_guard.py, which checks it before a request body is read). Not multi-user auth — one key for the whole API.
- **`API_READ_KEY`** (text; default empty) — Optional second, read-only key (needs API_KEY set too): it can ask questions and read documents, traces and jobs, but not upload, delete, re-index, tag, export, or read feedback/metrics — those need API_KEY (403 for the read key). For sharing the UI with people who should use the corpus but not change it.
- **`CORS_ALLOW_ORIGINS`** (list of text; default `[]`) — Extra browser origins allowed to call the API (JSON list). Inert in the primary paths (dev uses the Vite proxy, prod serves the frontend same-origin via StaticFiles) — a fallback for a contributor who points a separately-running frontend directly at this API instead of using the proxy.
- **`RATE_LIMIT_PER_MINUTE`** (integer; default `0`; ≥ 0) — Requests per minute each client may make that change something (POST, PATCH, DELETE on the API: questions, searches, uploads, feedback...) or that present a wrong API key; past it, 429 + Retry-After (api/rate_limit.py). Reads and job polls are never limited. 0 = off.
- **`FAILED_KEY_ATTEMPTS_PER_MINUTE`** (integer; default `30`; ≥ 0) — Wrong API keys each client may present per minute, whatever RATE_LIMIT_PER_MINUTE says (off by default, which left key guessing unlimited); past it, 429. 0 = off.
- **`MAX_CONCURRENT_QUESTIONS`** (integer; default `4`; ≥ 0) — Questions answered at once; past it a question gets 429 + Retry-After at once instead of holding a request thread for a minute (api/admission.py). 0 = no cap.
- **`MAX_CONCURRENT_SUMMARIES`** (integer; default `2`; ≥ 0) — Of those, how many may be summaries (POST /questions/summary\*), which run for minutes: without it four summaries held every slot and each question got 429. 0 = no separate cap.

## Logs, traces and feedback

What the server records about its own work and its users' questions.

- **`LOG_LEVEL`** (text; default `INFO`) — Verbosity of the application's own loggers (the "api" and "eval" namespaces). api/logging_config.py consumes this via dictConfig at app construction; without it the root logger's WARNING default silences every logger.info in the codebase (request logs, ingest progress, warmup notices).
- **`LOG_FORMAT`** ("text" or "json"; default `text`) — "json": one JSON object per log line (for a log shipper); "text" otherwise.
- **`TRACE_ENABLED`** (true/false; default true) — Per-query debug traces (retrieval candidates, rerank keep/drop decisions, the full generation prompt, stage timings) held in an in-memory ring buffer — same persistence tradeoff as IngestJobStore: lost on restart, fine for debugging a running instance. trace_max_retained caps memory use for large prompts/candidate lists; oldest traces are pruned first once the cap is exceeded.
- **`TRACE_MAX_RETAINED`** (integer; default `100`; > 0) — Traces kept in memory (oldest dropped first).
- **`TRACE_STORE_PATH`** (text; default empty) — Also keep traces in this sqlite file ("" = memory only): they then survive a restart and outlive the ring, so a cached answer's or a feedback row's trace id still opens. Holds prompts, so passages and what users typed. Oldest rows go past TRACE_STORE_MAX_ROWS. Never part of a backup.
- **`TRACE_STORE_MAX_ROWS`** (integer; default `5000`; > 0) — Oldest archived traces are dropped past this many.
- **`TRACE_LEG_RANKS`** (true/false; default false) — Record each candidate's rank on the dense and the sparse leg in its trace (which side found it). Free when fusion runs in the app; with Qdrant's own fusion it costs one more query per question, so it is off by default.
- **`RECHECK_TIME_BUDGET_SECONDS`** (number; default `45.0`; ≥ 0) — POST /query-log/recheck stops searching after this long and returns what it has (stopped_early): a proxy times a long request out (nginx's default is 60 s). 0 = no limit.
- **`FEEDBACK_ENABLED`** (true/false; default false) — Capture thumbs up/down on answers (POST /feedback, GET /feedback). False (default): no answer-feedback capture, matching every prior release. When true, POST /feedback (thumbs up/down, from ChatMessage.tsx) is enabled and PublicConfigResponse.feedback_enabled tells the frontend to render the buttons. Stores to feedback_store_path — its own sqlite3 file (stdlib, no new dependency), independent of job_store_backend: this feature works regardless of whether the operator has also opted the job store into sqlite persistence.
- **`FEEDBACK_STORE_PATH`** (text; default `./data/feedback.db`) — The feedback sqlite file (with FEEDBACK_ENABLED).
- **`FEEDBACK_MAX_ROWS`** (integer; default `50000`; > 0) — Oldest ratings are dropped past this many rows (~6 KB each at most).
- **`QUERY_LOG_ENABLED`** (true/false; default false) — A persistent log of every question and summary asked (text, workspace, how it went, best passage score), for "what do people ask that the documents don't cover?" (GET /query-log, full key). Off by default: it keeps what users typed. Its own sqlite file; never in a backup. Oldest rows go past QUERY_LOG_MAX_ROWS.
- **`QUERY_LOG_PATH`** (text; default `./data/query_log.db`) — The query log's sqlite file (with QUERY_LOG_ENABLED).
- **`QUERY_LOG_MAX_ROWS`** (integer; default `20000`; > 0) — Oldest query-log rows are dropped past this many.
- **`AUDIT_LOG_PATH`** (text; default empty) — Keep the audit events (deletes, uploads, re-indexes, restores...) in this sqlite file, newest AUDIT_LOG_MAX_ROWS, for GET /audit: log lines rotate away. "" = off.
- **`AUDIT_LOG_MAX_ROWS`** (integer; default `100000`; > 0) — Oldest audit rows are dropped past this many.

## Notifications

A webhook told when unattended work fails.

- **`NOTIFY_URL`** (text; default empty) — A webhook told when unattended work fails (api/notify.py); empty = off.
- **`NOTIFY_EVENTS`** (text; default empty) — Which events (comma-separated; empty = all): backup_failed, watch_gave_up, reindex_failed, url_refresh_failed, config_warning.
- **`NOTIFY_FORMAT`** ("json" or "text"; default `json`) — "json": {"text": ...} (Slack-compatible); "text": plain text + a Title header (ntfy).
- **`NOTIFY_ALLOW_PRIVATE`** (true/false; default false) — Allow a NOTIFY_URL on a private address (a service on your LAN).
- **`NOTIFY_MIN_INTERVAL_SECONDS`** (integer; default `3600`; ≥ 0) — At most one notification per event in this many seconds (repeats are counted).
