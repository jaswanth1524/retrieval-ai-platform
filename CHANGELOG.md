# Changelog

Every entry says whether existing documents need attention after upgrading:

- **Re-index**: documents must be re-uploaded or re-indexed to get the change.
- **Recommended**: it applies to new uploads; re-indexing brings old ones in line.
- **None**: it applies to the existing corpus as is.

The version is `version` in `pyproject.toml` (served as the API's OpenAPI version).

## Unreleased

Latency and features pass, then two improvement passes (fixes, security, accessibility,
features), then a follow-up retrieval and chunking pass (pass 8), then a local-model and
text-extraction pass (pass 9). **Re-index: recommended** — chunker version 8 (see
Changed). Every document
is listed as "older chunking" until it is re-indexed (↻ on its card, or "Re-index all
stale" with `RAW_DOCUMENT_DIR` set) or uploaded again; until then it keeps working as
before. No embedding change: the index stays compatible, and nothing is refused. The
re-index also picks up the parser fixes below (a mis-detected CSV delimiter, hidden HTML
text no longer indexed, an HTML-only email labelled by its subject, a web page's
navigation and a PDF's running headers left out).

**Upgrading with Compose:** the API's `/app/data` volume is now mounted by default, as
`docrag_data`. If you had uncommented the old `docrag_job_store` line, copy that volume
into the new one first (`docker run --rm -v <project>_docrag_job_store:/from -v
<project>_docrag_data:/to busybox cp -a /from/. /to/`).

### Changed
- **Chunker version 8** (pass 9). Every parsed document has ligature characters written
  out ("ﬁnance") and soft hyphens and zero-width spaces removed — they split a word in
  two for BM25 — and questions get the same clean-up (no NFKC: superscripts, fractions
  and full-width forms stay). PDF: a font whose `/ToUnicode` map contradicts the ligature
  glyphs its `/Differences` names is read by the glyphs (a published book came out as
  "speci!c", "!rst", "\"oor" in every extractor); words broken over a line end are
  rejoined, guided by the document's own words ("pur-chase" → "purchase", "cost-effective"
  kept); a book's running head with its page number ("26 Chapter 1: Prerequisites",
  "1.2 About Your App 27", alternating between left and right pages) is dropped, and a
  head that changes with each subsection loses its page number but stays as the page's
  section label ("1.2 About Your App"); a
  scanned page carrying only a stamp of text is OCR'd (its stamp kept when OCR can't run);
  a `|` inside a table cell no longer splits it. DOCX: text inside content controls,
  tracked insertions and simple fields is read (tracked deletions are not), a merged cell
  is read once, and a table with a header row labels its rows ("Grade: A; Days: 20").
  A binary file renamed to `.txt` is refused instead of indexed as control characters.
  Eval baselines re-measured: identical. **Recommended.**
- Pass 9: an uncited answer on a local model gets its citations by
  matching its sentences to the passages instead of a second generation
  (`CITATION_RETRY_MODE=auto`; `llm` restores the retry, `backfill`/`off` the others);
  hosted providers keep the LLM retry. **None.**
- **Chunker version 7** (pass 8). Short Markdown/HTML sections are no longer folded into
  their neighbours: a run of them becomes a chunk of its own, labelled by its first
  heading and cut once it reaches `MIN_SECTION_WORDS` (a policy's short "Receipts" and
  "Reimbursement" sections had been one chunk with its boilerplate introduction, cited as
  "About this policy", and the reranker scored the mileage rule 0.35 there and 0.64 on
  its own; a page of short FAQ answers had been one chunk under the last question). Only
  a scrap (a bare heading) still folds into the next section. PDF: a page's first or
  last line is matched across pages with its digits ignored only when it reads as a page
  number — "Article 3", "Notice period is 3 days" and "Total 180" were dropped from every
  page as if they were; a line repeated one in from the edge (a table's header row under
  the running header) keeps its first copy, where it had gone from every page, and isn't
  taken for that page's section label. HTML: a `<main>` or lone `<article>` is the page's content only
  when it holds at least half its text (a sidebar teaser `<article>` or a `<main>` with a
  welcome line had the real text in a `<div>` dropped), and `<header>`/`<aside>` inside
  `<main>`, `<article>` or `<section>` are kept. The token estimate used when the dense
  model's tokenizer can't be fetched counts CJK characters, digits and punctuation, so a
  chunk of unspaced Japanese or of figures no longer runs past the 512 tokens the model
  reads. Both retrieval baselines score the same as with version 6. **Recommended.**
- **Chunker version 6** (pass 7). HTML: only the page's content is indexed — `<main>`,
  or its single `<article>`, else the body without `<nav>`, `<header>`, `<footer>` and
  `<aside>` (kept inside an article; a page whose content area is empty is read whole);
  block elements end a line, so headings and list items no longer run together; a table
  is rows of `column: value` pairs (`| Grade: A; Budget: 500`); a charset declared only
  in a `<meta>` tag or the server's `Content-Type` (URL ingest) is honoured (a BOM still
  wins). PDF: lines repeated at the top or bottom of most pages (three pages or more) —
  running headers, footers, "Page 3 of 40" — are dropped; a page left empty by that is
  skipped, not sent to OCR. Both retrieval baselines score the same as with version 5.
- Pass 7: a summary too long for one call reads its parts side by side with a hosted
  provider (`SUMMARY_PARALLEL_PARTS`, 4; Ollama still one at a time), in document order
  and with every part's tokens counted. **None.**
- Pass 7: with `LLM_CONTEXT_WINDOW` unset, an OpenAI or Anthropic model's context window
  comes from LiteLLM's model table, so the prompt fitter and the citation retry's budget
  apply to them too (Ollama still uses `OLLAMA_NUM_CTX`; unknown models stay unfitted).
  Chinese, Japanese and Korean text is estimated at a token per character (it was a
  third of that), for prompts and summary parts. **None.**
- Pass 7: with `CONTEXT_NEIGHBOR_MAX_CHARS` set, the diversity filter keeps adjacent
  chunks (expansion no longer brings them back whole); the setting is in the eval
  fingerprint. Unchanged at the default (0). **None.**
- Pass 7: the chat keeps 100 conversations (was 20; history lives in IndexedDB now),
  with localStorage keeping a compact copy of the 20 most recent, and passage text kept
  in IndexedDB for the 20 most recently used — older conversations keep their citations.
  A notice says when the history fills. **None.**
- Pass 7: ∑, Compare and the palette's summary ask for an optional focus first. The
  corpus panel stays usable while an answer streams in another conversation. **None.**
- Pass 7: the cached answers and document listing are invalidated per workspace: an
  upload to one workspace no longer empties every workspace's answer cache. **None.**
- Pass 7: `MAX_CONCURRENT_SUMMARIES` (2) — summaries take one of their own slots as well
  as a question slot, so long summaries can't hold every slot. **None.**
- Internal: `api/pipeline.py` split into `api/context_expansion.py` (neighbour expansion)
  and `api/ingest_service.py` (`IngestService`), re-exported from `api.pipeline`; the
  corpus panel's card, bulk bar, tag editor and types moved to `components/corpus/`, and
  App's helpers and connection monitor to `appHelpers.ts` / `useConnectionMonitor`, its
  boot to `useAppBoot`, its workspace handling to `useWorkspaces`, backups to
  `useBackupActions`, the inspector's open/close rule to `useInspectorVisibility`, and the
  traces and feedback panels share `usePanelList`. No behaviour change (the OpenAPI
  document only gained the additions below).
- `GET /config` always reports the app's own settings (never a workspace's), so the UI
  boots even when the workspace it remembers was deleted.
- **Chunker version 5.** Table rows (CSV, XLSX and Markdown tables, which PDF tables are
  rendered as) are never split across chunks, and a table chunk is labelled with the
  rows it actually holds (`Rows 4-17`, not its whole 50-row group). A chunk that runs
  onto the next page records it (`page_end`, shown as "p.3–4" and given to the model as
  `page=3-4`). A short section is folded into its neighbour only on the same page (a
  short EPUB chapter was cited as the next chapter). Chunk ids come from the content,
  not the position, so an edit near the top of a document no longer gives every later
  passage a new id — a re-index keeps saved citations pointing at their passages. The
  handbook and sample retrieval evals score the same as with version 4.
- Uploads are parsed in a child process stopped after `PARSE_TIMEOUT_SECONDS` (300): a
  pathological file used to hold an ingest worker indefinitely, and parsing competed
  with questions for the GIL. `PARSE_TIMEOUT_SECONDS=0` restores in-process parsing.
- The document listing is re-read in full after `DOCUMENT_LISTING_TTL_SECONDS` (300; it
  was 30). Changes made by the API itself apply at once as before; the setting only
  bounds how long another process's writes go unseen.

### Added
- Pass 9: stream `stage` frames `waiting_for_model` (another question has the local
  model) and `reading` (`prompt_tokens`, and `eta_ms` once this server's prompt-reading
  speed is measured); `citation_backfilled` on answers, `done`, traces and the answer
  cache ("auto-cited" in the UI), `docrag_citation_backfills_total`. The UI shows the
  time a question has waited and the reading estimate, keeps "checking citations…" in
  view under text that has already streamed, explains each badge in words, and on a
  "not found" answer lists the passages it read and offers "Ask across all documents"
  (when it was scoped) and "Search the passages". Opt-in prompt trimming:
  `PROMPT_TOKEN_BUDGET`, `RERANK_RELATIVE_FLOOR` (trace drop reason `relative_cut`) and
  `GENERATION_HISTORY_ASSISTANT_CHARS`, all 0 (off). CLI: `docrag chat` (questions with
  the conversation as history), `ask --strict` (exit 3 when the documents don't answer),
  plain-language status notes and a progress tick every 15 s; MCP `ask` takes `history`.
  Eval: `answer` mode goes through `RagPipeline.answer` (history included), rows can carry
  `answer_contains` (`fact_hit_rate`, no judge model), rows are saved as they finish
  (`--resume`, `--limit`), and `make eval-retrieval|eval-answer|eval-ragas`. **None.**
- Pass 8: traces carry `condense_note` (why a follow-up wasn't rewritten, or its rewrite
  was cleaned or rejected; shown in the Trace tab) and the reranker model in `config`. The
  server warns at startup when `RERANKER_MODEL` is changed and `RERANK_MIN_SCORE` is not:
  the floor is on the default model's score scale. `eval/datasets/paraphrase_eval.jsonl`
  (questions worded unlike their passage) and six elliptical rows in
  `followup_eval.jsonl`. **None.**
- Pass 7: identical re-sends are left alone. A document uploaded (or watched, or fetched)
  again with the same bytes, under current chunking, finishes `done` with `unchanged:
  true` and nothing re-embedded; a watched file that was only touched takes its new time.
  Chunks carry `content_sha256`. **Recommended** (documents indexed before have no hash,
  so their next identical upload still re-indexes once).
- Pass 7: URL-ingested documents keep their source (`source_url`, with the server's
  ETag/Last-Modified): `source_urls` on `GET /documents`, a link on the card, and
  `POST /documents/{filename}/refresh` (and `docrag refresh`) to fetch again —
  conditionally, so an unchanged page costs a 304 and no job. Re-index keeps the source.
- Pass 7: `GET /documents/jobs` lists the jobs queued or running in a workspace, whoever
  started them; the corpus panel shows the ones this tab didn't start.
- Pass 7: the watched folder retries a file whose read or indexing job failed, with
  backoff (`WATCH_MAX_RETRIES`, `WATCH_RETRY_MAX_SECONDS`), and stops retrying a refused
  one until it changes. `GET /watch` reports its last pass and `POST /watch/scan` runs
  one now; the corpus panel shows both (`watch_enabled` on `/config`).
- Pass 7: `POST /search` narrows by `page_min`/`page_max`, `section` and
  `exclude_filenames`.
- Pass 7: `ANSWER_LANGUAGE` (`match`, or a language name) for answers and summaries.
- Pass 7: token usage the provider reported, per answer: `usage` on `QuestionResponse`,
  `SummaryResponse`, the `done` frame and the trace (generation and the citation retry;
  not condensing, query expansion or follow-ups).
- Pass 7: `/health/ready` reports `originals_kept`; the UI shows a banner when the index
  doesn't match the embedding setup (with the way out that works on this server), shows
  the server version in Settings, and offers a reload when the server is updated.
- Pass 7 UI: "Ask again" on a question left without an answer; a 429 is waited out with
  a countdown and asked again (up to three times); "weakly supported" on answers whose
  best passage only just cleared `RERANK_MIN_SCORE`; "not relevant" feedback on a single
  cited source (`citation_source_number`); "Ask about these" on documents; a jump to the
  latest turn; a question box that grows with the text; palette entries to read or
  summarize each document, copy the last answer and export every conversation (importing
  that file restores them all); ⌘⇧F search, ⌘\\ inspector, `]`/`[` between
  conversations and `?` for the list; clickable retrieval candidates; the trace's full
  id, request id, query variants, token usage and error; Traces and Feedback filters.
- Pass 7 CLI/MCP: `docrag read`, `delete`, `tags`, `reindex`, `refresh`, `jobs`,
  `workspaces`, `watch`, and `--tag` on `ask`/`search`; MCP `list_workspaces`, `tags`
  on `ask`/`search_passages`, `read_document(around_chunk_id=)`, and answers that list
  the exact sentences each citation rests on.
- Pass 7: briefs — one cited overview of 3 to `SUMMARY_MAX_DOCUMENTS` (5; at most 10,
  and never more than `SUMMARY_MAX_BATCHES`, since each document takes a part)
  documents, organised by theme: `mode: "brief"` on `/questions/summary*`, "Brief" for
  three or more selected documents in the corpus panel, `docrag summarize` with three
  or more files, and MCP `summarize`. **None.**
- Pass 7: a query log (`QUERY_LOG_ENABLED`, off by default; `QUERY_LOG_PATH`,
  `QUERY_LOG_MAX_ROWS` 20 000): one row per question or summary — text, workspace,
  outcome, best passage score — kept in its own sqlite file beyond the trace buffer.
  `GET /query-log` (full key; `?uncovered=true` for what the documents didn't answer)
  and `docrag queries`. **None.**
- Pass 7: document versions (`RAW_DOCUMENT_VERSIONS`, 0 = off; needs
  `RAW_DOCUMENT_DIR`): a re-upload keeps the previous original (up to that many);
  `GET /documents/{filename}/versions`, `POST /documents/{filename}/versions/{version}/restore`
  (full key; a job indexing those bytes like an upload, which keeps the version it
  replaces) and `docrag versions FILE [--restore VERSION]`. Deleting the
  document deletes its versions; backups hold only the current original. **None.**
- Pass 7: `QDRANT_QUANTIZATION=scalar` stores int8 copies of the dense vectors in RAM
  (about a quarter of the memory) and rescores the best `QDRANT_QUANTIZATION_OVERSAMPLING`
  (2.0) × limit candidates with the original vectors. Applied to an existing collection
  in place at startup, and removed again with `none`; in the eval fingerprint.
  **Re-index: none.**
- Pass 7 eval: answer-mode runs report `cited_relevant_rate`, `answered_rate`,
  `uncited_rate` and, on unanswerable rows, `refused_unanswerable_rate`; rows can carry
  a `history` and are scored on the rewritten follow-up, as the API retrieves it
  (`eval/datasets/followup_eval.jsonl`, handbook corpus).
- Pass 7 eval: datasets can hold unanswerable questions (`expect_no_context`), scored as
  `no_context_rate`; `eval/datasets/unanswerable_eval.jsonl` against the handbook corpus.
- Workspaces (`WORKSPACES_ENABLED`, off by default): separate corpora, one Qdrant
  collection (`<QDRANT_COLLECTION>__ws_<name>`) and originals folder each, chosen per
  request with `X-Workspace`; `GET/POST /workspaces`, `DELETE /workspaces/{name}` (empty
  only). The answer cache, the filename lock and the ingest sequencer are per workspace.
  The UI has a workspace menu in the corpus panel, and each conversation belongs to the
  workspace of its first question: choosing another workspace opens a fresh chat, and
  opening a conversation switches back to its workspace (the list names it). The
  workspace in use is held per browser tab. Answer feedback is kept per workspace too:
  `GET /feedback`, a backup and a restore cover only the request's workspace (the
  feedback file gains a `workspace` column in place, its rows in the default one), and
  `eval.harness from-feedback --store` takes `--workspace`. **Re-index: none** — the
  default workspace is the existing collection.
- `POST /documents/from-url` (`URL_INGEST_ENABLED`, off by default): the server fetches a
  document and indexes it like an upload; public addresses only unless
  `URL_INGEST_ALLOW_PRIVATE`, the checked address pinned for the connection, redirects
  re-checked, `MAX_UPLOAD_BYTES` and `URL_INGEST_TIMEOUT_SECONDS` apply. **Re-index: none.**
- A watched folder (`WATCH_DIR`, `WATCH_INTERVAL_SECONDS`, `WATCH_DELETE_MISSING`): new and
  changed files are indexed as they appear, into the default workspace or
  `WATCH_WORKSPACE` (created at startup when missing; a bad name, or workspaces off,
  stops startup naming the setting). **Re-index: none.**
- The `docrag` command (status, documents, ingest, add-url, ask, search, summarize,
  export, import) and an MCP server, `docrag-mcp`, both over the REST API. They are their
  own small package (`uv tool install ./cli`, `./cli[mcp]` for the server; httpx only,
  none of the server's dependencies); `python -m cli` / `python -m cli.mcp_server` run
  them from the repository. CI builds the wheel and runs it. **Re-index: none.**
- A multilingual profile: `SPARSE_LANGUAGE` (BM25 stemming) and `DENSE_PASSAGE_PREFIX`
  (e5-style models), both recorded with the collection and checked like the dense model;
  README "Multilingual documents" lists the supported models. A small Spanish/German/
  French fixture (`eval/corpus_multilingual/`, `eval/datasets/multilingual_eval.jsonl`,
  including English questions about the non-English documents) has its format and
  labels checked in CI (the labelled files and passages exist in the corpus); no model
  runs on it there and it has no baseline yet. **Re-index: none** for the defaults; switching to the profile is
  a new vector space (re-index).
- Whole-document summaries and comparisons: `POST /questions/summary` and
  `/questions/summary/stream` read every passage of one document (or two, to compare),
  in parts of `SUMMARY_BATCH_CHARS`, as cited notes combined into one answer; at most
  `SUMMARY_MAX_BATCHES` parts (a longer document is read at evenly spaced passages,
  reported as `chunks_read`/`chunks_total`), with `SUMMARY_MAX_TOKENS` for the answer.
  Same key, slot, rate limit, citation retry and answer cache as a question. The UI has
  ∑ on each document card, Compare for two selected documents, and a ⌘K command for a
  scoped document; Retry and Regenerate repeat the summary. **Re-index: none.**
- Suggested follow-up questions (`FOLLOWUPS_ENABLED`, off by default; `FOLLOWUP_COUNT`,
  `FOLLOWUP_MAX_TOKENS`): one extra LLM call after an answer that cited its sources,
  sent as `followups` on `/questions` and a `followups` frame after `done` on the
  stream, and shown as one-click questions under the latest answer. **Re-index: none.**
- Anthropic (Claude) generation: `LLM_PROVIDER=anthropic` or per question, with
  `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL` (default `claude-sonnet-5-5`). Optional like
  OpenAI; embeddings stay local. `GET /config` reports `anthropic_available`/
  `anthropic_model`. **Re-index: none.**
- `eval/harness.py from-feedback`: drafts a labeled retrieval dataset from answer
  ratings (the feedback store or a backup zip) — thumbs-up questions labeled with their
  cited documents, thumbs-down questions written to a review file. **Re-index: none.**
- Excel (`.xlsx`) and PowerPoint (`.pptx`) uploads (`openpyxl`, `python-pptx`). Each
  sheet is a page, its rows rendered like a CSV's and labelled with spreadsheet row
  numbers (`Costs · Rows 2-51`); each slide is a page titled by its slide title, with
  its tables and speaker notes. A slide or a sheet's row range is never merged into its
  neighbour, so even a one-line slide is cited as itself. `MAX_SPREADSHEET_CELLS`
  (200 000) caps a workbook; XML DTDs and zip bombs are refused as for DOCX and EPUB.
- `answer_status` gains `uncited` (an answer that cites none of its sources, even after
  the citation retry — it used to read as `answered`), and answers carry `truncated`
  when the model stopped at its token limit. The UI flags both ("no sources cited",
  "cut off").
- Metrics: `docrag_answers_total{status}`, `docrag_answers_truncated_total`,
  `docrag_citation_retries_total`, `docrag_feedback_total{rating}`,
  `docrag_llm_tokens_total{provider,kind}` (what the provider reports),
  `docrag_question_slots_in_use`, `docrag_ingest_backlog_bytes`/`_jobs`,
  `docrag_corpus_documents` and `docrag_build_info{version}`. `/health` reports the
  version. Under `LOG_FORMAT=json` each request gets one JSON access line with its
  request id (uvicorn's plain-text access line is silenced then).
- A cancelled or refused upload no longer empties the answer cache (only an ingest that
  started writing does). Passage search keeps adjacent passages it used to drop as
  neighbours of a better one (only answers expand neighbours back in).
- The README has a reverse-proxy guide (nginx, Caddy, `FORWARDED_ALLOW_IPS`,
  `--root-path`), curl examples and a Prometheus scrape config.
- EPUB and email (`.eml`) uploads. EPUB chapters are read in reading order, through the
  HTML parser; the chapter's position is its page. An email's headers (subject, sender,
  recipients, date) are a section of their own, then the plain-text body (or the HTML one
  when there is none). Attachments are not indexed. No new dependencies.
- `GET /config` lists the accepted upload types (`supported_extensions`); the UI's file
  picker, drop check and dropzone label follow it.
- Answers say how well they are grounded: `answer_status` (`answered`,
  `insufficient_context` — the model said the documents don't cover it and cited none —
  or `no_context` — nothing cleared the rerank bar) and `top_rerank_score`, on
  `/questions` and the stream's `done` frame, cached answers included. The UI marks such
  an answer "not in your documents".
- Pinned conversations: listed first and never dropped by the 20-conversation cap (at
  most 10 pins). Pin from the conversation row or ⌘K.
- Conversation search matches question and answer text, not just titles; a match shows
  where it was, and opening it scrolls to that turn.
- The source viewer can show a stored PDF original at the cited page (with
  `RAW_DOCUMENT_DIR` set, up to 50 MB), using the browser's own PDF viewer.
- Latency observability: `first_token_ms` (request to first streamed token) and
  `rerank_wait_ms` (time queued behind another question's rerank) in timings, traces and
  `docrag_question_stage_seconds`; `docrag_ingest_stage_seconds{stage=parse|chunk|embed|write}`.
- `INGEST_UPSERT_BATCH_SIZE` (default 256).
- `CONTEXT_NEIGHBOR_MAX_CHARS` (default 0, off): keep only that many characters of
  neighbour text on each side of a retrieved chunk. At `CHUNK_SIZE_TOKENS=96`, 300 cut
  about 1 s of prefill on qwen2.5 (M1) with the same answers and citations on 11
  questions; not measured at the default chunk size.
- Citations point at the passage an answer rests on: each cited source carries `spans`
  (`start`, `end`, `quote` within the chunk's text), matched from the answer's citing
  sentences without any model call (numbers match as digits or words: "45" finds
  "forty-five"). On `/questions` and the `done` frame (the `sources`
  frame, sent before the answer, has none). The viewer marks the passage and the hover
  preview quotes it instead of the chunk's opening.
- Corpus panel: read a document from its card (the viewer from its first passage);
  select several to delete, tag or re-index at once; filter by tag; sort by name, upload
  time or size (remembered).
- Other conversations can be opened, started, imported or deleted while an answer
  streams; it keeps writing to the conversation it was asked in, and the open one holds
  sending until it finishes.
- An unsent question is kept per conversation, across switches and reloads.
- An empty conversation offers example questions about the newest documents, or an
  upload button when there are none.
- The search panel keeps its query and results when another panel is opened.
- "API unreachable" retries on its own (2 s doubling to 30 s) and offers Retry now; a
  trace that failed to load can be retried from the inspector.
- `make check` runs the backend and frontend gates CI runs; `make golden` regenerates
  the pinned snapshots.
- `RATE_LIMIT_PER_MINUTE` (default 0, off): requests per minute per client that change
  something (questions, searches, uploads, feedback, deletes…) or carry a wrong API key;
  past it, 429 + `Retry-After`. Reads and upload progress polls are never limited.
  Counted in `docrag_rate_limited_total`.

### Faster
- The reranker scores candidates shortest first, so each batch pads to similar lengths:
  about 22% faster over 50 candidates (jina int8, M1). The int8 model's scores depend
  slightly on which passages share a batch, so low-scoring candidates can trade places
  (measured: top result unchanged on 10 of 10 questions, handbook eval metrics identical);
  that was already true of the order fusion happened to produce.
- Search returns only the six payload fields a candidate is built from; neighbour
  expansion only two.
- A tag-scoped question gets its tags' stored spellings from one facet call, alongside
  embedding, instead of a full scan of the collection whenever the listing cache was cold.
- Repeated query text (Regenerate, Retry) reuses its embedding.
- Condensing a follow-up sends the last four messages, with long answers clipped, instead
  of the whole history.
- The manual-fusion fallback sends both retrieval legs in one Qdrant call.
- Ingest writes 256 points per upsert instead of 64, so the filename lock is held through
  fewer round trips.
- The UI boots with `/config`, `/access` and the document list in parallel (no `/health`
  first) and says "Connecting…" instead of "Upload a document" until the list arrives;
  the question box can be typed in meanwhile.
- Built JS and CSS ship precompressed (`.br`, `.gz`) and are served with a year of
  caching under `/assets/`; `index.html` is revalidated. About 70% fewer bytes on a cold
  load.
- Uploads: two files are sent at a time, but an accepted file no longer holds its slot
  while it is indexed; job polling drops back to 1 s whenever a job moves.
- Fewer re-renders while an answer streams (memoized panels, stable handlers), a
  reloaded chat stays scrolled to the bottom as it renders, a hovered citation prefetches
  the viewer, and switching conversations is a transition.
- `GET /export` streams its zip as it is written (bounded memory, nothing spooled to the
  container's tmpfs), and reads feedback a page at a time.
- After an upload, delete or tag change the document listing re-reads only the
  documents that changed instead of scanning the whole collection; a restore reads it
  once rather than once per document.
- The listing finds stored originals with one directory read; an upload with a known
  size is read in one call (it used to be held twice); ingest holds sparse vectors as
  arrays; feedback pruning runs only past the cap.
- The corpus panel re-renders only the card an upload poll changed; a long conversation
  lays out only the turns near the viewport.

### Fixed
- Pass 9: long questions on a local model timed out before the first token.
  `LLM_REQUEST_TIMEOUT_SECONDS` reached httpx as the longest silence allowed (LiteLLM
  turns an `httpx.Timeout` back into one number for Ollama), and Ollama is silent while
  it reads the prompt: an 8.5k-token prompt failed at 60 s on an M1. A local call's
  timeout now adds the prompt's estimated tokens over `LLM_PREFILL_TOKENS_PER_SECOND`
  (40, or the slower rate measured), and Ollama's non-streamed calls are streamed and
  joined so the timeout bounds a silence, not the whole answer; the same prompt answers
  (first token at 79 s). The timeout message names the time it allowed. **None.**
- Pass 9: a second question asked while Ollama answered another timed out without ever
  being read — Ollama answers one request at a time, and the queued one's timeout ran.
  Calls wait in DocRAG for `OLLAMA_MAX_PARALLEL` (1) places, and their timeout starts
  when they are sent; both questions answer. **None.**
- Pass 9: Ollama streams reported no token usage (`usage` was always empty); streams now
  ask for it. **None.**
- Pass 9: an uncited refusal ("The documents do not mention X.") was `uncited`, and
  went to a second generation asking for citations — on the weak-match path, against a
  prompt that had said not to cite. It is `insufficient_context`, and a weak-match
  refusal is never retried. An answer with a caveat ("Mileage is thirty cents per
  kilometre, but the document does not give the currency [1]", "… — it does not specify
  the days") is `answered`, not a refusal; "does not include" and a refusal that lists
  the sections it had are refusals. **None.**
- Pass 9: the eval's `answer` mode ignored a row's `history`, so a follow-up was asked
  as a standalone question. **None.**
- Pass 8: a question the documents answer was refused ("I do not have enough
  information…") whenever the reranker scored every passage under `RERANK_MIN_SCORE`.
  The floor is on one model's score scale: "what are foundational models" against a
  passage starting "Foundation models: You have trained your own model…" scored 0.006
  with `Xenova/ms-marco-MiniLM-L-6-v2` and 0.154 with the default jina reranker, so the
  answer depended on which side of 0.15 a score fell. Now the `RERANK_FALLBACK_COUNT`
  (3) best-ranked passages go to the model anyway, and the prompt tells it they match
  only weakly: answer only if they state the answer, else say the documents don't cover
  it. Such an answer carries `weak_matches: true` (`QuestionResponse`, the `done`
  frame, the answer cache) and is marked "weak match" in the UI; a refusal is still
  `insufficient_context`. `RERANK_FALLBACK_COUNT=0` restores the instant refusal. A
  question nothing clears the bar for now costs one LLM call. Measured on 42
  answerable and 12 unanswerable questions: gpt-5-mini answered 41/42 (jina) and 40/42
  (MiniLM), up from 37 and 35; qwen2.5 37/42 with either, up from 37 and 35; every
  configuration still refused 12/12. Refusals that cite the sources they looked at are
  recognised in more of the ways gpt-5-mini and qwen2.5 phrase them (a refusal after
  ", but", sentences that only say what the sources are about, a trailing run of bare
  markers). **None.**
- Pass 8: traces carry `weak_matches` (shown in the Trace tab), and `docrag ask` and the
  MCP `ask` tool say when an answer rests on weak matches. **None.**
- Pass 8: a line copying a source's header ("[1] filename=…; page=…; section=…"), which
  qwen2.5 appended to answers despite the prompt, is removed from the answer. **None.**
- Pass 8: a second question in a conversation was often answered "not enough
  information" although the documents held the answer. Follow-ups such as "Explain in
  more detail", "Give me more details", "List all the steps", "Elaborate on the second
  point" or "How about driving a private car?" were judged standalone and searched as
  typed; with no topic, nothing cleared `RERANK_MIN_SCORE`. Such follow-ups (words like
  more, details, explain, second, steps; openings like "and …", "what about …") are now
  rewritten from the conversation first, and the raw follow-up is searched next to the
  rewrite (fused with the same RRF `k=60`), so a rewrite that lost what was asked no
  longer loses its matches. The rewrite is cleaned (an echoed "Standalone question:"
  label, a preamble, quotes, a second line) and not used when it ran into
  `CONDENSE_MAX_TOKENS`, is far longer than a question, or answers instead of asking.
  The answering prompt shows the question as asked plus how it was read. Measured on
  `followup_eval.jsonl`: hit@3 0.5 → 1.0. **None.**
- Pass 8: an exchange answered "not enough information" (no citation) is left out of the
  history sent with the next question, by the server and by the UI: the rewrite folded
  the refusal into the next search ("What if they miss it?" became "… miss the key
  points?") and the model refused again. **None.**
- Pass 8: an answer that cites sources only to say they don't answer the question ("The
  provided documents do not contain enough information to determine how many sick days
  contractors get [1][2].") read as `answered`: no "not in your documents" badge, and it
  went back as history. Its `answer_status` is `insufficient_context` now, so it is
  flagged and left out of history like an uncited refusal. An answer that gives
  something after what is missing ("…, but days carry over [1]", or a later cited
  sentence) is still `answered`. **None.**
- Pass 8: a summary or comparison answer went back as history at up to 4000 characters
  (half the history budget); it is clipped to 1500. New chat on an empty conversation
  kept the document scope set on it ("Ask about these"), so the first question of the
  new chat was searched in those documents only; it starts unscoped now. **None.**
- Pass 8: a trace whose answer said the documents don't cover the question was recorded
  `ok`; it is `insufficient_context` now (and counted as uncovered by the query log).
  **None.**
- Pass 7: earlier answers went back to the model as history with their `[n]` markers,
  which numbered the previous turn's sources: the model could copy a stale number onto
  an unrelated source. They are stripped from history now.
- Pass 7: a German, Spanish, French, Portuguese or Italian follow-up ("Und wie lange
  dauert das?") was never rewritten into a standalone query (the check knew English words
  only), and a "not enough information" answer in those languages cost a citation retry
  and read as uncited.
- Pass 7: the citation retry could push the prompt past the context window (it adds the
  first answer and a reminder to a prompt already fitted): it drops the history, then
  clips the first answer, then skips the retry.
- Pass 7: a summary stream left as `done` arrived was not cached.
- Pass 7: with `RAW_DOCUMENT_DIR` unset, the index-mismatch message still advised
  "export, delete every document, import" — which restores nothing without originals.
  It names only what works, and a backup says what it lacks (`originals_missing`,
  `complete` in the manifest, `X-DocRAG-Originals-Missing`).
- Pass 7: the watched folder marked a refused or failed file as sent and never tried it
  again until it changed or the process restarted.
- Pass 7: a conditional URL fetch answered 304 was handled as a redirect.
- Pass 7: a comparison (or brief) too long for one call answered that the documents
  don't contain enough information: each part shows one document while the task names
  all of them, and the model replied "NOTHING RELEVANT" for every part. The notes prompt
  now says the other documents are read separately.
- Pass 7 UI: "Preparing backup… the download starts when it is ready" stayed up for
  seconds after the download had started, and a backup slower than 4 s lost the note
  while it was still being built. It now stays until the backup is ready and goes the
  moment the download starts, fails or is cancelled.
- Pass 7: the watched folder gives up at once on a file that can't be parsed (it is
  retried when it changes), instead of re-reading it `WATCH_MAX_RETRIES` times.
- Pass 7 UI: the document list was read at startup only — documents added by the watched
  folder, the CLI or another tab appeared after a reload. It is re-read when the tab
  comes back into view (at most every 30 s) and when the connection returns.
- A question sent (Enter) while the page was still starting up was dropped without a
  word, left in the box; it is now sent as soon as DocRAG is ready, with a "Sending as
  soon as DocRAG is ready…" note meanwhile (not when sending waits for another
  conversation's answer). The watched folder's "lists no files at all" / "could not be
  read" warnings are logged once when deletions start being held back, not every
  interval.
- A one- or two-word search (`POST /search`, the Search panel, `docrag search`, MCP
  `search_passages`) often found nothing although documents held the words: the
  cross-encoder scores a bare keyword low even against a passage containing it
  ("laptop": 0.11 for the IT equipment policy, which it still ranks first), under the
  0.15 `RERANK_MIN_SCORE` an answer's sources must clear. When nothing clears it, a
  search now returns the closest passages with `weak_matches: true`, and the panel, the
  CLI and the MCP tool say they are only the closest. Answers are unchanged: they never
  use passages under the bar. **Re-index: none.**
- The stronger multilingual profile (`intfloat/multilingual-e5-large`) failed to load
  from a fresh download ("External data path escapes model directory"): its weights are
  a separate `model.onnx_data` file, huggingface_hub 1.x links each file into a shared,
  hash-sharded blob store, and ONNX Runtime 1.27 refuses external data outside the
  directory the model resolves into. Model files are now hard-linked into the model's
  own directory after download (same files, no extra disk) — dense, sparse and reranker
  models alike. Measured on `eval/datasets/multilingual_eval.jsonl` the default, light
  and stronger profiles all ingest and retrieve (chunk hit@1 0.83, MRR 0.92, recall@3
  1.0 each; a small set). **Re-index: none.**
- Corpus panel: the URL field's placeholder no longer runs out of room, and the workspace
  label sits on its own line so Delete doesn't wrap alone. A card's actions move under
  its file name when they don't fit beside it (with ∑ added, names broke mid-word). On a
  phone, the composer's provider menu spans the control row instead of running off the
  right edge. `docrag status` names the model of the provider in use (it always showed
  Ollama's).
- Review of the roadmap features, all with regression tests:
  - **Watched folder (security):** only regular files are read — a symbolic link (to
    `/proc/self/environ`, `/dev/zero`), FIFO or device is skipped, and the read is bounded
    and refuses a file swapped for a link after the listing. `WATCH_DELETE_MISSING`
    deletes nothing in a pass that couldn't read the whole folder (an unreadable
    subfolder, a dropped mount) or that lists nothing at all. A file is read once it has
    been unchanged for one interval (not half-copied), documents are stamped with the
    file's mtime and re-indexed whenever it differs (a `cp -p` replacement was missed),
    and a name differing only by case from a stored one is refused as uploads are. Watched
    documents indexed before this are re-indexed once.
  - **URL ingest:** checks the indexing queue before connecting and the declared size
    before the body (503 + Retry-After), at most 4 fetches at once (429), one deadline for
    the whole fetch, bad or non-ASCII URLs are a 400 not a 500 (hosts IDNA-encoded), IPv4
    is tried first with fallback to the next address, NAT64/6to4/IPv4-mapped addresses are
    checked by the IPv4 inside, and logged URLs drop their query string. The deadline
    now covers the DNS lookup too, and is exact: when it passes, the connection is shut
    down under whatever read was waiting (a single slow read could run past it by up to
    the whole timeout again). A name
    the remote server picked (its `Content-Disposition`, or the path of a URL it
    redirected to) that differs from the URL's own and is already indexed is a 409
    instead of replacing that document; giving a `filename` replaces it on purpose, and
    re-adding the same URL still replaces its own document.
  - **Workspaces:** traces and ingest jobs belong to the workspace they ran in (404 from
    another); deleting a workspace with work in flight is a 409 (the check and the delete
    are now one step: an upload or delete into the workspace meanwhile is a 409, and an
    import defers those documents, where they could slip in between), a deleted
    workspace is never recreated by a request or job that found it just before the
    delete (404 "No workspace named …"; only creating it makes it), the watched folder's
    `WATCH_WORKSPACE` can't be deleted while it is configured (409), leftover temp files no
    longer block it, and a re-created workspace stores originals again; `/health/ready`
    ignores `X-Workspace` (it could reveal workspace names without a key); a collection
    name containing `__ws_` is refused at startup; `FEEDBACK_MAX_ROWS` applies per
    workspace. The UI clears the previous workspace's cards, selection and search on a
    switch, sends each upload, delete, tag change, re-index and URL ingest to the
    workspace it started in, refreshes the workspace list after Settings or a key change,
    and falls back to the default workspace when the current one was deleted elsewhere.
    Arrowing through the closed workspace menu (Windows, Linux) switches once, when the
    keys stop, on Enter or when focus leaves, not at every workspace passed.
  - **Answers:** a streamed answer shows as finished at `done`, with follow-ups arriving
    afterwards (Regenerate, feedback and the next question no longer wait for them), and
    is cached at `done`; Anthropic temperatures above 1 are a 400 before any work;
    equivalent `SPARSE_LANGUAGE` values ("", "English") are no longer refused.
  - **Summaries:** refused with the same 409 as questions when the index was built for
    another embedding setup (they read stored text, but a mismatched index is rebuilt
    before it answers anything); notes mentioning "nothing relevant" in passing are kept, a cut passage
    leaves room for its header, comparing a document with no readable text is a 400
    naming it, and a short document's unused parts go to the other one.
  - **Feedback** records its `kind` (`question`, `follow_up`, `summary`); `from-feedback`
    labels only questions, reads the store read-only, and defaults to the default
    workspace (`--all-workspaces` for all).
  - **CLI:** files with the same name in different folders are uploaded once and the
    rest reported, an error after the answer finished no longer discards it, `export`
    never truncates an existing backup, `import` exits 1 when anything failed, local
    errors are `error:` lines (no tracebacks), global flags work after the subcommand,
    output can't crash a cp1252 console, 429 is retried, `add-url` no longer treats the
    page as the server address, and MCP `read_document` output is capped. CI's wheel
    smoke test imports the installed wheel, not the source checkout. `ingest` keeps up to
    8 files indexing at once (the server indexes them side by side) instead of one after
    another; `ingest`, `add-url` and `import` stop waiting after `--timeout`
    (default 3600 s, 0 = no limit) and report what is still indexing; `ingest` and
    `add-url` honour `--json`. A cancelled MCP `ask` or `summarize` closes its stream
    (within one keepalive, 15 s), so the answer stops on the server and frees its
    question slot. The dependency audit (CI, `make check`) covers the `mcp` extra.
- A client that left a streamed answer kept the pipeline running to the end of the
  answer (Starlette never closes a plain iterator): the question slot and the LLM stayed
  busy until garbage collection. The same for `/original` downloads, whose file stayed
  open. Both now stop as soon as the client leaves.
- A per-request provider that isn't configured (`openai` with no key) is refused with a
  400 before any retrieval, instead of failing after the rerank — on the stream, after
  the sources had been sent.
- Backup restores no longer fail with a 500 halfway through on a damaged, encrypted or
  unsupported zip member: that member is listed in `rejected` and the rest goes on.
- A huge `citation_source_number` on `/feedback`, or a 400-digit `start`/`end` on
  `/documents/{f}/content`, was a 500 (now a 422).
- Neighbour expansion could miss a neighbouring chunk next to leftover copies from an
  interrupted re-index.
- The UI kept a remembered Ollama selected while Ollama was down (every question
  failed) instead of switching to a provider that could answer; the stored preference
  is kept for when it is back.
- A failed first document listing (a 500, a timeout) was shown as an empty corpus,
  disabling the question box under "Upload a document to start"; it is retried now.
- Settings opened before the server's config loaded blocked every shortcut and popped
  up by itself later; clearing a conversation didn't reach other tabs (their next save
  brought it back); bulk actions could include documents hidden by the filter, a bulk
  delete dropped focus, and a bulk re-index refreshed the listing once per document;
  a re-index or restore blocked uploads; toasts shown while the API was unreachable
  were lost; a large backup restore timed out after two minutes, and a backup download
  can now be cancelled (the same palette command); the viewer's "passage changed" note
  stayed up for a later citation that was found; drafts could be pruned before the
  history had loaded; switching between conversations of the same length kept the
  scroll position; an inspector opened on a wide window became a focus-trapping drawer
  when the window narrowed.
- The UI notices when the server goes away after boot (it said "api ok" for the rest of
  the session): `/health` is checked every 30 s while the page is visible, and the
  browser's offline event is believed at once. Renames, pins, deletes and switches made
  in other conversations while an answer streams are saved straight away. The theme
  follows the OS live until one is picked, and a pick in one tab applies to the others.
- `QDRANT_GRPC_PORT` is used by the API's Qdrant client (it always dialled 6334). A
  default LLM provider that isn't configured is warned about at startup. The release
  gate fails within minutes for a tag on a commit CI never ran for, instead of after an
  hour, and `## 0.2.0` no longer matches a `## 0.2.0-rc.1` CHANGELOG heading.
- The answer stream is read by the SSE rules: a proxy that rewrites line endings, splits
  an event's data over several lines or drops the space after `data:` no longer makes
  frames vanish, and a last frame with no blank line after it is kept.
- A pasted question cut at the length limit says so; ⌘K's arrow keys skip commands that
  can't run; a message from another day shows its date (exports always do).
- The PDF "Original" view broke after switching to Passages and back (its blob URL was
  freed on the switch), and stayed on "Loading" when switched away mid-download. The
  download is now cancelled when the viewer closes.
- With the markdown renderer unavailable, answers showed `[1](#docrag-cite-1)`.
- A citation marker with an absurdly long number (`[999…9]`) turned a finished answer
  into a server error.
- The CSV parser could pick a letter as the delimiter of a one-column file of prose.
- Re-indexing one document read its original into memory before checking the indexing
  queue had room.
- Restoring feedback could fail the whole import on a `NaN` timestamp and held the file
  twice in memory; rows the API couldn't have stored are now skipped.
- Retried uploads bypassed the two-at-a-time send limit; finished upload cards piled up
  for the session; a hovered citation's preview stayed up after switching conversation
  from the keyboard.
- The eval harness stopped at the first non-document file (`.DS_Store`) in a corpus
  directory.
- "Show all" (and Load earlier) in the source viewer threw the reader far from the
  passage they were reading; it now stays in place.

### Security
- Pass 9: Pillow 13.0.0, for the two dozen advisories against 12.3.0 fixed only in 13.
  fastembed (every release, 0.9.0 included) caps pillow<13, so the lock overrides it
  (`[tool.uv] override-dependencies`): fastembed uses Pillow for its image models only,
  and DocRAG loads text models. The PDF, PPTX and OCR parsers run on Pillow 13. The
  audit's ignore list (and its test) is gone, having no entries left. **None.**
- Pass 8: pypdf 6.21.2 (at least 6.20.1), for CVE-2026-113792 to CVE-2026-113795 in
  6.19.0. PDF text extraction is unchanged on the golden documents. **None.**
- Pass 7: aiohttp 3.14.5 (through LiteLLM) and certifi 2026.10.19, for advisories
  published since the last bump. Eight Pillow 12.3.0 advisories are fixed only in Pillow
  13, which no fastembed release accepts yet: the dependency audit skips them, listed
  with that reason in `.github/ci/pip-audit-ignore.txt`, and a test fails once the
  lockfile moves past Pillow 12 so the skips are removed then.
- Pass 7: the query log keeps what users typed — off by default, read only with the full
  key, capped, and never in a backup; kept versions are earlier originals on disk, removed
  with the document. SECURITY.md says both.
- Pass 7: SECURITY.md covers URL ingest, the watched folder, workspaces (not access
  control), the MCP server's key and every hosted provider's data flow.
- Behind `uvicorn --root-path`, the request guard compared the prefixed path and let
  every guarded route through unchecked (no early 401, no body cap, no rate limit, open
  API docs). It now matches the path the routes see.
- Section labels and file names get the same `[n] filename=` rewriting as source text
  in the prompt (a heading could still start a fake source on the real header's line).
- HTML text hidden from readers (`hidden`, `aria-hidden="true"`, inline
  `display:none`/`visibility:hidden`) is no longer indexed — it was a channel for
  instructions aimed at the model that nobody reviewing the page would see. Applies to
  HTML, EPUB and HTML email.
- `GET /export` runs one at a time (a second is a 409) and counts against
  `RATE_LIMIT_PER_MINUTE`: each export reads every stored original.
- Narrowed: `citation_source_number` at most 1000, `/content` `start`/`end` at most
  1 000 000.
- Qdrant's anonymous telemetry is turned off in the Compose file.
- OCR renders each page within a pixel budget (`OCR_MAX_PAGE_PIXELS`, 40 MP): a tiny PDF
  declaring a page hundreds of inches wide asked for gigabytes. Large pages are rendered
  at a lower resolution, and a page too large even at 72 dpi is skipped.
- The XML DTD check for XLSX, PPTX and EPUB also catches UTF-16 and UTF-32 encoded
  members.
- The UI is served with a Content-Security-Policy (scripts, styles, fonts and requests
  from its own origin only; frames only for the viewer's PDF; no other site may frame
  it) and `X-Frame-Options: DENY`; every response carries `X-Content-Type-Options:
  nosniff` and `Referrer-Policy: no-referrer`. A frontend served from another origin is
  not covered by this policy. Checked in Chromium and WebKit (`frame-ancestors 'self'`:
  with `'none'` WebKit refused the viewer's own PDF frame).
- Runtime base image bumped (`python:3.12-slim-bookworm`, same Python 3.12.15): its
  `perl-base` had three fixable CRITICAL CVEs (CVE-2026-13221, CVE-2026-42496,
  CVE-2026-8376), caught by the new CI image scan.
- Request lists are capped per item, not only in count: `trace_id` 64 characters, file
  names 1024, tags 128 (on `/questions*`, `/search` and `/feedback`). Longer values are a
  422 where they used to be accepted.
- A document's text can no longer open a fake `[n] filename=…` source in the prompt:
  header-like text anywhere in a source is rewritten as `(ref) filename=…` in the prompt
  only (with gpt-5-mini, a planted `[2] filename=payroll.xlsx` line was cited as source
  2; rewritten, it is cited as part of the document it is in), and section labels are
  kept to one line. One sentence was added to the system prompt saying where a source
  starts.

### Accessibility
- Pass 7: "Load earlier/later" in the viewer moves focus to the first passage it loaded
  once the button goes; the composer's scope menu closes when focus leaves it; on touch
  screens Enter adds a line (send with the button) and the keyboard hint is hidden; the
  viewer fills the screen on phones, and dialogs fit between mobile browser bars
  (`dvh`, with a `vh` fallback).
- The citation preview closes with Escape, and the inline `[n]` markers behave like the
  citation cards (no reopening when focus returns from the viewer, no stuck preview on
  touch). Answer headings start at `<h3>`, below the page's own; wide tables scroll
  inside the answer; long words wrap. The feedback filters' emoji are no longer read
  aloud; held-down shortcuts don't repeat. The citation being previewed points screen
  readers at the preview (`aria-describedby`, `role="tooltip"`). Light theme `--accent`
  is darker (it was 4.2:1 on `--sunk`); the contrast test now checks every text colour.
- Focus returns to the control that opened an inline editor (rename, edit question,
  tags) and to the next card after a delete; the viewer's focus trap includes the PDF
  frame and skips hidden controls; "Skip to question" link; pinned conversations are
  announced as pinned; focus outlines show in Windows high-contrast mode; failures and
  confirmations are announced from separate live regions; passage search announces how
  many it found.

### Development
- Pass 7: `tests/test_frontend_contract.py` checks that every field the frontend's API
  types read exists in the OpenAPI schema. CI loads the embedding and reranker models
  inside the hardened compose container (read-only filesystem, dropped capabilities)
  and embeds and reranks one query; the release starts the arm64 image (imports the
  app) before pushing it, and builds the `docrag-cli` wheel and sdist, attaching them
  to the release and publishing to PyPI only when the `PUBLISH_CLI` repository variable
  is `true` (trusted publishing, which needs setting up on PyPI first).
- Pass 7: the eval baselines record `rrf_k`, the retrieval limits, the query-expansion
  count, the reranker batch size, `sparse_language` and `dense_passage_prefix`, so a
  change to any of them is a refused comparison rather than a silent pass (both
  baselines re-measured: identical aggregates). `make check` imports ragas, as CI does. Tests for the
  restore guards and URL-fetch edge cases; two wall-clock assertions made robust.
- `make check` runs what CI runs, coverage floor, dataset validation and dependency
  audits included. CI can be started by hand; lint and typecheck run once (on 3.12);
  dependency review skips with a notice while the repository's Dependency graph is
  off; the OCR image is also booted (its engine imported) and scanned; e2e fails on a
  test that only passed on retry.
- An enabled store (`RAW_DOCUMENT_DIR`, sqlite jobs, feedback) that can't write its
  directory stops startup with the setting named, instead of a 500 on first use.
  `FORWARDED_ALLOW_IPS` is documented in `.env.example`.
- `tests/golden/openapi.json` pins the API schema; its test names any breaking change
  before the diff (`make openapi` regenerates it).
- `httpx`, `numpy` and `grpcio`, imported by `api/` directly, are declared.
- Frontend tests are type-checked in CI (`npm run typecheck:tests`); backend coverage
  must stay at or above 92%.
- The eval fingerprint also records `rrf_k`, the retrieval limits, the query expansion
  count, the reranker batch size and the machine (the machine is never a refusal).
- Releases run only for a tag on a commit that is on `main` and passed CI, and both the
  amd64 and arm64 images must pass the fixable-CRITICAL scan before anything is pushed
  (only amd64 was scanned). CI scans its image with the same gate, reviews the
  dependencies a pull request adds, and runs weekly on `main` so new advisories against
  the pinned versions are noticed without a push.

## 0.1.0 — 2026-10-10

The first release. It collects everything below: four review passes and the work before
them. If you are upgrading from a development build:

- **Re-index: Recommended.** `CHUNKER_VERSION` is now 4 (section labels are capped,
  Markdown code fences no longer split sections, CSV sections name spreadsheet rows, and
  a long filename can no longer shrink the chunks). Every existing document is listed
  as stale once; with `RAW_DOCUMENT_DIR` set, the corpus panel's "Re-index all" applies it.
  Documents now also record which chunk settings made them, so changing
  `CHUNK_SIZE_TOKENS`, the overlap or the dense model lists them as stale too.
- **Changed the embedding model or `EMBEDDING_MODEL_TAG`?** Listing, exporting, tagging
  and deleting documents keep working, and `/health/ready` reports `index_compatible:
  false`. Questions and uploads are refused until the index matches again: download a
  backup (`GET /export`), delete every document, then `POST /import` it (an emptied
  collection is rebuilt for the new model), or delete everything and upload again.
- **Docker Compose now publishes the API on `127.0.0.1` only.** Set `API_BIND=0.0.0.0` in
  `.env` to reach it from other devices, and set `API_KEY` when you do.

### Changed in the fourth review pass
- Docker Compose publishes the API on `127.0.0.1` only. Set `API_BIND=0.0.0.0` in `.env`
  to open it to other devices, and set `API_KEY` when you do.
- An upload into an index built for another embedding setup is refused up front (409)
  instead of being accepted and failing in the background.
- A collection whose recorded dense model differs from `DENSE_EMBEDDING_MODEL` is treated
  as mismatched even when `EMBEDDING_MODEL_TAG` was left unchanged.
- With `API_KEY` set, `/docs`, `/redoc` and `/openapi.json` need the key too.
- `LLM_PROVIDER` must be `ollama`, `openai` or `openai_compatible`, and
  `CHUNK_OVERLAP_TOKENS` must be below `CHUNK_SIZE_TOKENS`; both are checked at startup.
- The UI fonts are bundled, so the app makes no third-party requests.

### Fixes from the fourth review pass
- Rebuilding an empty mismatched collection re-checks under the lock, so a collection
  that gained points meanwhile is never deleted.
- A misspelled embedding or reranker model gives a named error; at warmup it is logged
  and the app still boots.
- An ingest job that crashed outside its own error handling is marked failed instead
  of staying in progress.
- `POST /import` rejects one document with a bad tag instead of stopping partway.
- The raw-store startup sweep only removes its own temporary files.
- Traces label a chunk cut by `max_context_chunks` as `context_limit`.
- LiteLLM no longer fetches its model price list from GitHub at import (Docker, tests).
- UI:
  - An answer keeps streaming into the conversation it was asked in, even when another
    tab renames or deletes it; a delete from another tab applies once the answer ends.
  - Adjacent citations (`[1][2]`) are clickable.
  - A corrupt timestamp in imported history can't crash the page.
  - The Settings sliders can't build a request the server rejects.
  - History sent with a question leaves out unanswered and repeated questions.
  - Deleted tags drop out of conversation scopes, and the scope picker stops at the
    server's limits.
  - An answer streaming in a hidden tab is saved, and a closed IndexedDB connection
    is reopened.
  - Enter in the command palette always runs the highlighted row.
  - Failure toasts stay for 10 s, and focus, labels and contrast are fixed in a few
    places.
- Eval: a passage-only dataset row searches every file. The RAGAS runner fails when any
  metric scored nothing, and writes NaN as `null`.

### Fixes
- Deleting a document no longer removes a newer upload of it that won the race for the
  document's lock (409 instead); a failure removing the stored original after the
  points are gone is logged, not a 500.
- Embedding a document no longer holds the document's lock: tag changes and deletes of
  it are not blocked for minutes behind a large re-index, and a cancel now works
  throughout embedding, not just before its first batch.
- A Markdown heading of any length no longer multiplies a document's chunks (a
  2000-character heading made 4351 chunks); `# comments` inside fenced code blocks are
  not headings; a window overlap larger than the budget no longer advances one sentence
  at a time; CSV sections are labelled with spreadsheet row numbers.
- `/documents` listing: concurrent requests share one scan of the collection, and a
  scan slower than the 30 s TTL still caches.
- A wrong `QDRANT_DENSE_VECTOR_SIZE` is named at boot and at the first upload, not
  reported as "retry the upload"; Qdrant's own error statuses (a rejected key, a 5xx)
  are 503s with a fixed message, and one failed hybrid query no longer switches
  server-side hybrid search off for the process.
- A client that leaves mid-answer now closes the HTTP stream to the model server, so
  Ollama stops generating for nobody.
- The tokenizer is retried after a failed fetch (it stayed on the word heuristic for the
  life of the process); documents record which counter sized them.
- With stored originals, `Report.pdf` and `report.pdf` (or two Unicode forms of one
  name) can no longer share one original: the second upload is a 409.
- Question tags match stored tags regardless of case; request ids reach the worker
  threads of query expansion and embedding; restored feedback keeps its original
  order; a busy `/search` is not counted as a question; two `X-API-Key` headers are a
  400; leftover `.upload-*` temp files are swept; a reranker that can't load names
  `RERANKER_MODEL` instead of the library's error text.
- Frontend: a clicked trace shows that trace (not the latest answer's); Cancel aborts an
  upload still being sent and large uploads get a time budget that grows with their
  size; a drop on the corpus panel during an upload is kept; other tabs' renames,
  deletes and long histories no longer revert, resurrect or truncate each other; a bad
  imported conversation can't crash the app on every load (and there is a recovery
  page if anything does); light-theme text meets 4.5:1 contrast; ⌘ shortcuts no longer
  take Ctrl+K from text fields on a Mac; IME Enter no longer submits; sizes show as
  KB/MB/GB; a rating that failed to send can be sent again; downloads time out
  instead of leaving "Download a backup" disabled; many smaller accessibility and layout
  fixes.
- The Docker image reported version 0.0.0; it now reports the release's.

### Features
- Passage search in the UI (rail panel and palette): the passages a question would be
  answered from, with scores, scoped like questions and working with no LLM reachable.
- Stale documents are detected from the chunk settings they were made with, not only
  the chunker version.

### Operations
- Release: a fixable CRITICAL vulnerability stops the release before anything is
  pushed; images are published for linux/amd64 and linux/arm64; the CHANGELOG must have
  this version's section. Compose can run a released image (`DOCRAG_VERSION`,
  `DOCRAG_PULL_POLICY=always`).
- CI: pushes to main no longer cancel each other; actions are pinned by commit; the
  image is booted through compose with its hardening and a real Qdrant, and must report
  the pyproject version; the OCR image build runs on main; the model cache is keyed on
  what is downloaded; the eval baseline refuses a run made with different ranking
  settings and its results are uploaded.
- Compose waits 60 s for running ingests when stopping.
- Dependabot version-update config removed; dependencies are bumped by hand.

## History before 0.1.0

What the development builds each added, newest first.

## Second review pass (`feat/review-pass-2`)

Re-index: **None**. No chunker or embedding change. New settings default to today's
behaviour except the PDF caps (`MAX_PDF_PAGES=2000`, `MAX_OCR_PAGES=100`).

### Fixes
- A client that disconnects mid-question no longer frees its question slot while the
  retrieval and rerank it started keep running; an abandoned stream stops before the
  rerank or the LLM call.
- An upload queued before a `DELETE` of the same document no longer indexes it back.
- A delete whose stored original can't be removed still invalidates cached answers.
- A filename over 255 bytes is refused at upload; one already indexed no longer makes
  `GET /documents` fail with `RAW_DOCUMENT_DIR` set (Python 3.12).
- Citations written `[1, 2]` or `[1-3]` count as citations (no needless retry call).
- PDFs over `MAX_PDF_PAGES` are refused; OCR stops after `MAX_OCR_PAGES` scanned pages.
- `/content` without paging returns at most 2000 chunks (`total_chunks` stays exact).
- `/export` skips an original deleted while it runs instead of failing.
- Model and Qdrant errors no longer put internal exception text in responses, job
  errors or traces.
- UTF-16/UTF-32 text files (with a BOM) are decoded correctly.
- Frontend: a partly streamed answer whose connection fails is marked incomplete and
  kept out of history; dropping a file outside the corpus panel no longer navigates
  away; no duplicate empty "New chat" after a reload; iOS layout uses `100dvh`; the
  composer no longer pops the keyboard after each answer on touch screens; the
  conversation delete confirm's Cancel button is visible in the light theme.

### Features
- `POST /search`: ranked passages without an answer (works with the LLM down).
- `DELETE /documents/jobs/{id}`: cancel an ingest job before it writes anything; the
  corpus panel's indexing cards have a Cancel button (also stops an upload still waiting
  for room in the queue).
- `POST /import`: restore a `/export` backup (originals, tags, upload times, feedback).
- `GET /access`: which key the caller holds; the UI hides what the read-only key
  can't do.
- `ALLOWED_REQUEST_PROVIDERS`: restrict per-question provider switching.
- `X-Request-ID` on every response, in every log line and in traces; `LOG_FORMAT=json`.
- UI: clickable `[n]` citations, download the original document, restore from the
  palette, a Feedback panel, a tab-title badge (and optional notification) when an
  answer finishes in the background, the answering model recorded per answer, the
  chosen provider remembered.

### Performance
- While streaming, only the markdown block still growing is re-parsed per token.
- The corpus panel and conversation list no longer re-render on every token.
- History writes to IndexedDB are coalesced; uploads go two at a time.
- The markdown renderer is loaded after the first paint, not with the app: the first
  load fetches 280 KB of JavaScript instead of 435 KB (88 KB gzipped instead of 135 KB).
- The chat header no longer re-renders on every token.
- Screen readers hear answer progress from one status line for the thread instead of
  one region per message (and "Answer ready." only for an answer they watched arrive).

### Operations
- Docker: the runtime stage is plain `python:3.12-slim` (no uv); base images pinned
  by digest; compose rotates container logs (10 MB x 3).
- Release workflow: GHCR image with provenance and SBOM, cosign signature, Trivy scan,
  GitHub release from this file, on a `v*` tag.
- CI runs the handbook retrieval eval against its baseline; tests time out after 120 s
  and report coverage.

### Code layout
- `api/main.py` split into `api/routes/*`, `api/ingest_jobs.py`, `api/sse.py` and
  `api/error_handlers.py`; `api/generation.py` into `api/llm.py`, `api/prompting.py`
  and `api/citations.py`; format parsers into `api/parsers/*`. No API change.
- Removed `retrieve_candidates`, `rerank_candidates` and `filename_chunk_counts`, used
  only by tests (which now exercise the production paths); test settings are built by
  `tests/factories.py`.
- Frontend: `useCommands` (palette list and shortcuts), `ConfirmInline` (the three
  inline delete/clear confirms), one metadata record per document in `useCorpus`.

### Evaluation
- `relevant_passages` labels (resolved to chunk ids at run time) and nDCG@k.
- The handbook corpus gained three distractor documents and is scored with small
  chunks, so it is no longer saturated (reranked chunk hit@1 0.95, fused 0.74).

## Full review pass (after PR #12)
Re-index: **None**. The BM25 IDF modifier is enabled on an existing collection in place
the first time the API touches it — no re-embedding. Tags start empty.

#### Retrieval
- BM25 is scored with Qdrant's IDF modifier, so rare terms outweigh common ones (it was
  term frequency only). Existing collections are migrated automatically.
- The reranker scores `filename › section` + text, the same string that is embedded.
- With `OLLAMA_NUM_CTX` or `LLM_CONTEXT_WINDOW` set, the prompt is fitted to the model's
  window (lowest-ranked sources first, then the oldest history); traces mark what was dropped.

#### Features
- Answer cache for repeat questions (`ANSWER_CACHE_SIZE`, `ANSWER_CACHE_TTL_SECONDS`).
- Document tags: `PATCH /documents/{filename}/tags`, `tags` on `/questions*`, tag chips in
  the corpus panel and the scope popover.
- `LLM_PROVIDER=openai_compatible` for vLLM, llama.cpp, LM Studio and similar servers.
- Chat: edit a question into a forked conversation, JSON import, conversation search,
  per-conversation scope, copy with citation footnotes, `/` and Esc shortcuts, OS theme
  by default, stopped and cached answers labelled, trace status in the traces panel.

#### Fixes
- The app no longer crashes on first load over plain HTTP on a LAN address.
- Browser history: old answers' passage text is no longer saved (it filled the quota),
  conversations are pruned by last use, two tabs merge instead of overwriting, and a
  question in flight survives a reload.
- Bulk re-index and large drops no longer hand back job ids that already 404.
- A queued re-index, or a slower older upload, can no longer overwrite a newer upload.
- `DELETE`/`/content`/`/original` refuse names that were never stored as given.
- Password-protected PDFs are a 400; one broken PDF page no longer fails the document.
- CSV cells with line breaks, and HTML comments, are handled correctly.
- Streaming answers survive proxies (keepalives, no buffering); `/health` answers under load.

#### Security and operations
- Unauthorized or oversized requests are rejected before their body is read.
- Provider error messages no longer reach clients verbatim; URLs are shown without credentials.
- Qdrant is published on 127.0.0.1 only; `QDRANT_API_KEY` is supported.
- Feedback rows are capped (`FEEDBACK_MAX_ROWS`).
- Compose: restart policy, API memory cap, a real Qdrant readiness check, `UV_EXTRAS=ocr`.
- CI: `uv sync --locked`, the image is booted, server-only Qdrant tests, eval typechecked.

#### Also in this release
- Read-only API key (`API_READ_KEY`); `MAX_CONCURRENT_QUESTIONS` (429 past it); uploads
  retry by themselves when the indexing queue is full.
- `GET /export` backup zip; paged `/documents/{filename}/content` (`around`, `start`/`end`).
- Chat history in IndexedDB; the prompt fitter drops sources before history.
- CJK sentence splitting and cutting of over-long unspaced "words" (re-upload CJK or
  base64-heavy documents to benefit); PDF pages with no drawn lines skip the table finder.
- A Qdrant collection deleted under a running API is recreated instead of failing.
- `GET /documents` is served from a cache between corpus changes.
- The API container is read-only with no capabilities; CI covers Python 3.13 and
  `ruff format`; `drop_reason` is declared an extensible enum.

#### Evaluation
- The harness measures the retrieval path the API runs, per stage (`--arm`), against a
  committed fixture corpus and baseline; mismatched baselines are refused.
- RAGAS can judge with a local model (`--llm`, `--embeddings fastembed`); all-NaN runs fail.

## PR #12 — review follow-ups

Re-index: **Recommended** to use re-index at all: every chunk now records its
`chunker_version`, so documents indexed earlier show as stale.

- Re-index from stored originals (`POST /documents/{filename}/reindex`, bulk variant).
- Narrow-screen drawers, lazy-loaded dialogs, accessibility pass, upload retry/dismiss.
- LLM timeout messages, OCR failure handling, DOCX zip-bomb guard, shutdown handling.

## PR #10 / #11 — latency pass

Re-index: **Recommended** (`CHUNKER_VERSION` 3: token counts exclude special tokens).

- Ollama via `ollama_chat/` (keep-alive now works), int8 jina reranker by default,
  prompt and context trimming, rerank score cache, Playwright E2E in CI.
