import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import ConfigPanel from '../../src/components/ConfigPanel';
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
    rerank_min_score: 0.3,
    max_upload_bytes: 52428800,
    rerank_top_k_limit: 50,
    max_context_chunks_limit: 20,
    llm_temperature_max: 2.0,
    ...overrides,
  };
}

const EMPTY: QuestionOverrides = { rerankTopK: null, maxContextChunks: null, llmTemperature: null };

describe('ConfigPanel', () => {
  it('renders RRF k and Fused top N as plain read-only cards with no stepper buttons', () => {
    render(<ConfigPanel config={makeConfig()} overrides={EMPTY} onOverridesChange={() => {}} />);

    expect(screen.getByText('60')).toBeInTheDocument();
    expect(screen.getByText('RRF k')).toBeInTheDocument();
    expect(screen.getByText('50')).toBeInTheDocument();
    expect(screen.getByText('Fused top N')).toBeInTheDocument();
    expect(screen.queryByLabelText('Increase RRF k')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Increase Fused top N')).not.toBeInTheDocument();
  });

  it('renders Rerank top K and Context chunks as steppers defaulting to config values', () => {
    render(<ConfigPanel config={makeConfig()} overrides={EMPTY} onOverridesChange={() => {}} />);

    expect(screen.getByLabelText('Increase Rerank top K')).toBeInTheDocument();
    expect(screen.getByLabelText('Decrease Rerank top K')).toBeInTheDocument();
    expect(screen.getByLabelText('Increase Context chunks')).toBeInTheDocument();
  });

  it('increments rerank_top_k and clamps at the server limit', async () => {
    const onOverridesChange = vi.fn();
    render(
      <ConfigPanel
        config={makeConfig({ rerank_top_k: 8, rerank_top_k_limit: 8 })}
        overrides={{ ...EMPTY, rerankTopK: 8 }}
        onOverridesChange={onOverridesChange}
      />,
    );

    await userEvent.click(screen.getByLabelText('Increase Rerank top K'));

    expect(onOverridesChange).toHaveBeenCalledWith(
      expect.objectContaining({ rerankTopK: 8 }),
    );
  });

  it('decrements context chunks and clamps at 1', async () => {
    const onOverridesChange = vi.fn();
    render(
      <ConfigPanel
        config={makeConfig()}
        overrides={{ ...EMPTY, maxContextChunks: 1 }}
        onOverridesChange={onOverridesChange}
      />,
    );

    await userEvent.click(screen.getByLabelText('Decrease Context chunks'));

    expect(onOverridesChange).toHaveBeenCalledWith(
      expect.objectContaining({ maxContextChunks: 1 }),
    );
  });

  it('steps temperature by 0.1 and clamps at llm_temperature_max', async () => {
    const onOverridesChange = vi.fn();
    render(
      <ConfigPanel
        config={makeConfig({ llm_temperature: 0.2 })}
        overrides={EMPTY}
        onOverridesChange={onOverridesChange}
      />,
    );

    expect(screen.getByText('0.2')).toBeInTheDocument();

    await userEvent.click(screen.getByLabelText('Increase Temperature'));

    expect(onOverridesChange).toHaveBeenCalledWith(
      expect.objectContaining({ llmTemperature: 0.3 }),
    );
  });

  it('reset button zeroes all three overrides', async () => {
    const onOverridesChange = vi.fn();
    render(
      <ConfigPanel
        config={makeConfig()}
        overrides={{ rerankTopK: 3, maxContextChunks: 2, llmTemperature: 1.0 }}
        onOverridesChange={onOverridesChange}
      />,
    );

    await userEvent.click(screen.getByRole('button', { name: 'Reset to defaults' }));

    expect(onOverridesChange).toHaveBeenCalledWith(EMPTY);
  });

  it('disables all stepper and reset buttons when disabled prop is set', () => {
    render(
      <ConfigPanel config={makeConfig()} overrides={EMPTY} onOverridesChange={() => {}} disabled />,
    );

    expect(screen.getByLabelText('Increase Rerank top K')).toBeDisabled();
    expect(screen.getByLabelText('Decrease Context chunks')).toBeDisabled();
    expect(screen.getByLabelText('Increase Temperature')).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Reset to defaults' })).toBeDisabled();
  });

  it('still supports the System info toggle', async () => {
    render(<ConfigPanel config={makeConfig()} overrides={EMPTY} onOverridesChange={() => {}} />);

    expect(screen.queryByText('Qdrant collection')).not.toBeInTheDocument();
    await userEvent.click(screen.getByText('System info'));
    expect(screen.getByText('Qdrant collection')).toBeInTheDocument();
  });
});
