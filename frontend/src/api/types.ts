export interface HealthResponse {
  status: string;
}

export interface PublicConfigResponse {
  qdrant_collection: string;
  dense_embedding_model: string;
  sparse_embedding_model: string;
  reranker_model: string;
  embedding_model_tag: string;
  llm_provider: string;
  llm_model: string;
  llm_temperature: number;
  openai_model: string;
  openai_available: boolean;
  ollama_available: boolean;
  rrf_k: number;
  dense_retrieval_limit: number;
  sparse_retrieval_limit: number;
  fused_top_n: number;
  rerank_top_k: number;
  max_context_chunks: number;
  rerank_min_score: number;
  max_upload_bytes: number;
  rerank_top_k_limit: number;
  max_context_chunks_limit: number;
  llm_temperature_max: number;
}

export interface QuestionOverrides {
  rerankTopK: number | null;
  maxContextChunks: number | null;
  llmTemperature: number | null;
}

export interface DocumentIngestResponse {
  filename: string;
  sections_parsed: number;
  chunks_ingested: number;
  collection_name: string;
}

export type IngestJobState = 'queued' | 'parsing' | 'embedding' | 'indexing' | 'done' | 'failed';

export interface DocumentJobAcceptedResponse {
  job_id: string;
  filename: string;
  state: 'queued';
}

export interface DocumentJobStatusResponse {
  job_id: string;
  filename: string;
  state: IngestJobState;
  chunks_total: number;
  chunks_done: number;
  error: string | null;
  result: DocumentIngestResponse | null;
}

export interface DocumentListResponse {
  filenames: string[];
}

export interface CitationResponse {
  source_number: number;
  filename: string;
  page: number;
  section: string;
  chunk_id: string;
  text: string;
}

export interface TimingsResponse {
  embed_ms: number;
  search_ms: number;
  rerank_ms: number;
  generate_ms: number;
  total_ms: number;
}

export interface QuestionResponse {
  answer: string;
  sources: CitationResponse[];
  timings?: TimingsResponse | null;
}

export type LlmProvider = 'ollama' | 'openai';
