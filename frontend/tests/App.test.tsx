import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
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
            dense_embedding_model: 'BAAI/bge-small-en-v1.5',
            sparse_embedding_model: 'Qdrant/BM25',
            reranker_model: 'BAAI/bge-reranker-v2-m3',
            embedding_model_tag: 'fastembed:BAAI/bge-small-en-v1.5',
            llm_provider: 'ollama',
            llm_model: 'llama3.1:8b',
            openai_model: 'gpt-4o-mini',
            openai_available: false,
            rrf_k: 60,
            dense_retrieval_limit: 50,
            sparse_retrieval_limit: 50,
            fused_top_n: 50,
            rerank_top_k: 8,
            max_context_chunks: 6,
            max_upload_bytes: 52428800,
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

  it('persists the toggled theme to localStorage under the docrag-theme key', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/health')) return jsonResponse({ status: 'ok' });
        if (url.endsWith('/config')) {
          return jsonResponse({
            qdrant_collection: 'docrag_documents',
            dense_embedding_model: 'BAAI/bge-small-en-v1.5',
            sparse_embedding_model: 'Qdrant/BM25',
            reranker_model: 'BAAI/bge-reranker-v2-m3',
            embedding_model_tag: 'fastembed:BAAI/bge-small-en-v1.5',
            llm_provider: 'ollama',
            llm_model: 'llama3.1:8b',
            openai_model: 'gpt-4o-mini',
            openai_available: false,
            rrf_k: 60,
            dense_retrieval_limit: 50,
            sparse_retrieval_limit: 50,
            fused_top_n: 50,
            rerank_top_k: 8,
            max_context_chunks: 6,
            max_upload_bytes: 52428800,
          });
        }
        throw new Error(`unexpected fetch: ${url}`);
      }),
    );
    const setItemSpy = vi.spyOn(Storage.prototype, 'setItem');

    render(<App />);
    await screen.findByText('API online');

    await userEvent.click(screen.getByTestId('theme-toggle'));

    // Assert against the post-click DOM state rather than a predicted pre-click
    // value — document.documentElement persists across tests within this file, so
    // the theme a fresh <App/> mounts into isn't reliably predictable here.
    const themeAfterToggle = document.documentElement.dataset.theme;
    expect(setItemSpy).toHaveBeenCalledWith('docrag-theme', themeAfterToggle);
  });

  it('still flips the visible theme when localStorage.setItem throws', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith('/health')) return jsonResponse({ status: 'ok' });
        if (url.endsWith('/config')) {
          return jsonResponse({
            qdrant_collection: 'docrag_documents',
            dense_embedding_model: 'BAAI/bge-small-en-v1.5',
            sparse_embedding_model: 'Qdrant/BM25',
            reranker_model: 'BAAI/bge-reranker-v2-m3',
            embedding_model_tag: 'fastembed:BAAI/bge-small-en-v1.5',
            llm_provider: 'ollama',
            llm_model: 'llama3.1:8b',
            openai_model: 'gpt-4o-mini',
            openai_available: false,
            rrf_k: 60,
            dense_retrieval_limit: 50,
            sparse_retrieval_limit: 50,
            fused_top_n: 50,
            rerank_top_k: 8,
            max_context_chunks: 6,
            max_upload_bytes: 52428800,
          });
        }
        throw new Error(`unexpected fetch: ${url}`);
      }),
    );
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new DOMException('denied', 'SecurityError');
    });

    render(<App />);
    await screen.findByText('API online');

    const initialTheme = document.documentElement.dataset.theme;
    await userEvent.click(screen.getByTestId('theme-toggle'));

    expect(document.documentElement.dataset.theme).not.toBe(initialTheme);
  });

  it('shows the unreachable banner when health check fails', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));

    render(<App />);

    expect(await screen.findByText('API unreachable')).toBeInTheDocument();
    expect(screen.getByText(/Cannot reach the DocRAG API/)).toBeInTheDocument();
    expect(screen.getByRole('alert')).toHaveTextContent(/Cannot reach the DocRAG API/);
  });
});
