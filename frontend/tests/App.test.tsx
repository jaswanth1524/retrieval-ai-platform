import { render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from '../src/App';

afterEach(() => {
  vi.unstubAllGlobals();
});

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } });
}

describe('App', () => {
  it('renders the shell and reaches "API online" when health+config succeed', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/health')) return jsonResponse({ status: 'ok' });
        if (url.endsWith('/config')) {
          return jsonResponse({
            qdrant_collection: 'docrag_documents',
            embedding_provider: 'local',
            dense_embedding_model: 'BAAI/bge-small-en-v1.5',
            sparse_embedding_model: 'Qdrant/BM25',
            reranker_model: 'BAAI/bge-reranker-v2-m3',
            embedding_model_tag: 'fastembed:BAAI/bge-small-en-v1.5',
            llm_provider: 'ollama',
            llm_model: 'llama3.1:8b',
            openai_model: 'gpt-4o-mini',
            rrf_k: 60,
            dense_retrieval_limit: 50,
            sparse_retrieval_limit: 50,
            fused_top_n: 50,
            rerank_top_k: 8,
            max_context_chunks: 6,
          });
        }
        throw new Error(`unexpected fetch: ${url}`);
      }),
    );

    render(<App />);

    expect(await screen.findByText('API online')).toBeInTheDocument();
    expect(screen.getByText('DocRAG')).toBeInTheDocument();
    expect(screen.getByTestId('question-textarea')).toBeInTheDocument();
  });

  it('shows the unreachable banner when health check fails', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));

    render(<App />);

    expect(await screen.findByText('API unreachable')).toBeInTheDocument();
    expect(screen.getByText(/Cannot reach the DocRAG API/)).toBeInTheDocument();
  });
});
