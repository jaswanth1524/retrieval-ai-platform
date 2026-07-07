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
  openai_model: string;
  openai_available: boolean;
  rrf_k: number;
  dense_retrieval_limit: number;
  sparse_retrieval_limit: number;
  fused_top_n: number;
  rerank_top_k: number;
  max_context_chunks: number;
  max_upload_bytes: number;
}

export interface DocumentIngestResponse {
  filename: string;
  sections_parsed: number;
  chunks_ingested: number;
  collection_name: string;
}

export interface CitationResponse {
  source_number: number;
  filename: string;
  page: number;
  section: string;
  chunk_id: string;
  text: string;
}

export interface QuestionResponse {
  answer: string;
  sources: CitationResponse[];
}

export type LlmProvider = 'ollama' | 'openai';
