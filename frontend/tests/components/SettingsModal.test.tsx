import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import SettingsModal from '../../src/components/SettingsModal';
import type { PublicConfigResponse, QuestionOverrides } from '../../src/api/types';

function makeConfig(overrides: Partial<PublicConfigResponse> = {}): PublicConfigResponse {
  return {
    qdrant_collection: 'docrag_documents',
    dense_embedding_model: 'BAAI/bge-small-en-v1.5',
    sparse_embedding_model: 'Qdrant/BM25',
    reranker_model: 'jinaai/jina-reranker-v2-base-multilingual',
    embedding_model_tag: 'fastembed:BAAI/bge-small-en-v1.5',
    llm_provider: 'ollama',
    llm_model: 'llama3.1:8b',
    llm_temperature: 0,
    openai_model: 'gpt-4o-mini',
    openai_available: false,
    ollama_available: true,
    rrf_k: 60,
    dense_retrieval_limit: 50,
    sparse_retrieval_limit: 50,
    fused_top_n: 50,
    rerank_top_k: 8,
    max_context_chunks: 6,
    rerank_min_score: 0.15,
    max_upload_bytes: 52428800,
    rerank_top_k_limit: 50,
    max_context_chunks_limit: 20,
    llm_temperature_max: 2.0,
    ...overrides,
  };
}

const EMPTY: QuestionOverrides = { rerankTopK: null, maxContextChunks: null, llmTemperature: null };

function setup(props: Partial<Parameters<typeof SettingsModal>[0]> = {}) {
  const merged = {
    config: makeConfig(),
    overrides: EMPTY,
    onOverridesChange: vi.fn(),
    onClose: vi.fn(),
    onSaved: vi.fn(),
    ...props,
  };
  render(<SettingsModal {...merged} />);
  return merged;
}

describe('SettingsModal', () => {
  beforeEach(() => localStorage.clear());

  it('takes slider bounds from the server, never hardcoded values', () => {
    // A slider that can reach a value the backend rejects is a 400 waiting to happen,
    // and rerank_top_k_limit is capped by the live fused_top_n.
    setup({ config: makeConfig({ rerank_top_k_limit: 5, max_context_chunks_limit: 9, llm_temperature_max: 1.0 }) });

    expect(screen.getByTestId('settings-rerank-top-k')).toHaveAttribute('max', '5');
    expect(screen.getByTestId('settings-max-context-chunks')).toHaveAttribute('max', '9');
    expect(screen.getByTestId('settings-temperature')).toHaveAttribute('max', '1');
  });

  it('falls back to the server defaults when no override is set', () => {
    setup({ config: makeConfig({ rerank_top_k: 8, max_context_chunks: 6 }) });

    expect(screen.getByTestId('settings-rerank-top-k')).toHaveValue('8');
    expect(screen.getByTestId('settings-max-context-chunks')).toHaveValue('6');
  });

  it('reports slider changes as overrides', () => {
    const props = setup();

    // Range inputs are dragged, not typed, so drive the change event directly.
    fireEvent.change(screen.getByTestId('settings-max-context-chunks'), { target: { value: '11' } });

    expect(props.onOverridesChange).toHaveBeenCalledWith({ ...EMPTY, maxContextChunks: 11 });
  });

  it('rounds temperature to one decimal to survive repeated drags', () => {
    // Binary float drift (0.1 + 0.2 !== 0.3) otherwise accumulates into a value the
    // server's own ge/le bounds would still accept but that renders as 0.7000000001.
    const props = setup();

    fireEvent.change(screen.getByTestId('settings-temperature'), { target: { value: '0.7000000001' } });

    expect(props.onOverridesChange).toHaveBeenCalledWith({ ...EMPTY, llmTemperature: 0.7 });
  });

  it('keeps the Access section with the API key field', () => {
    // Deliberately not in the design: a deployment with API_KEY set is unusable from
    // the browser without it, and every /documents call 401s until it is filled in.
    setup();

    expect(screen.getByText('Access')).toBeInTheDocument();
    expect(screen.getByTestId('settings-api-key')).toHaveAttribute('type', 'password');
  });

  it('persists the API key only on save', async () => {
    const props = setup();

    await userEvent.type(screen.getByTestId('settings-api-key'), 'secret-key');
    // Typing alone must not commit it — Cancel has to actually cancel.
    expect(localStorage.getItem('docrag-api-key')).toBeNull();

    await userEvent.click(screen.getByTestId('settings-save'));

    expect(localStorage.getItem('docrag-api-key')).toBe('secret-key');
    expect(props.onSaved).toHaveBeenCalled();
    expect(props.onClose).toHaveBeenCalled();
  });

  it('resets overrides back to the server defaults', async () => {
    const props = setup({ overrides: { rerankTopK: 20, maxContextChunks: 3, llmTemperature: 1.4 } });

    await userEvent.click(screen.getByTestId('settings-reset'));

    expect(props.onOverridesChange).toHaveBeenCalledWith(EMPTY);
  });

  it('marks an unreachable provider', () => {
    setup({ config: makeConfig({ ollama_available: false, openai_available: true }) });

    expect(screen.getByText('unreachable')).toBeInTheDocument();
    expect(screen.getByText('key configured')).toBeInTheDocument();
  });

  it('is a modal dialog that closes on Escape and on backdrop click', async () => {
    const props = setup();

    const dialog = screen.getByTestId('settings-modal');
    expect(dialog).toHaveAttribute('role', 'dialog');
    expect(dialog).toHaveAttribute('aria-modal', 'true');

    await userEvent.keyboard('{Escape}');
    expect(props.onClose).toHaveBeenCalledTimes(1);

    await userEvent.click(screen.getByTestId('settings-backdrop'));
    expect(props.onClose).toHaveBeenCalledTimes(2);
  });

  it('moves focus into the dialog on open', () => {
    setup();

    expect(screen.getByTestId('settings-modal').contains(document.activeElement)).toBe(true);
  });
});
