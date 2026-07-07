import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import ProviderSelector from '../../src/components/ProviderSelector';
import type { PublicConfigResponse } from '../../src/api/types';

function makeConfig(overrides: Partial<PublicConfigResponse> = {}): PublicConfigResponse {
  return {
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
    ...overrides,
  };
}

describe('ProviderSelector', () => {
  it('renders both provider options with their model names', () => {
    render(
      <ProviderSelector config={makeConfig()} value="ollama" onChange={() => {}} />,
    );

    expect(screen.getByRole('option', { name: /Local \(Ollama · llama3\.1:8b\)/ })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: /OpenAI \(gpt-4o-mini\)/ })).toBeInTheDocument();
  });

  it('disables the OpenAI option and shows a hint when no key is available', () => {
    render(
      <ProviderSelector config={makeConfig({ openai_available: false })} value="ollama" onChange={() => {}} />,
    );

    expect(screen.getByRole('option', { name: /OpenAI/ })).toBeDisabled();
    expect(screen.getByTestId('provider-openai-hint')).toBeInTheDocument();
  });

  it('enables the OpenAI option and hides the hint when a key is available', () => {
    render(
      <ProviderSelector config={makeConfig({ openai_available: true })} value="ollama" onChange={() => {}} />,
    );

    expect(screen.getByRole('option', { name: /OpenAI/ })).toBeEnabled();
    expect(screen.queryByTestId('provider-openai-hint')).not.toBeInTheDocument();
  });

  it('calls onChange when a provider is selected', async () => {
    const onChange = vi.fn();
    render(
      <ProviderSelector config={makeConfig({ openai_available: true })} value="ollama" onChange={onChange} />,
    );

    await userEvent.selectOptions(screen.getByTestId('provider-select'), 'openai');

    expect(onChange).toHaveBeenCalledWith('openai');
  });
});
