import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import TraceBrowser from '../../src/components/TraceBrowser';
import { api } from '../../src/api/client';

vi.mock('../../src/api/client', async () => {
  const actual = await vi.importActual<typeof import('../../src/api/client')>('../../src/api/client');
  return {
    ...actual,
    api: { ...actual.api, listTraces: vi.fn(), getTrace: vi.fn() },
  };
});

const listTracesMock = vi.mocked(api.listTraces);
const getTraceMock = vi.mocked(api.getTrace);

afterEach(() => {
  listTracesMock.mockReset();
  getTraceMock.mockReset();
});

const SUMMARY = {
  trace_id: 't1',
  created_at: 0,
  question: 'What is DocRAG?',
  mode: 'stream' as const,
  status: 'ok' as const,
  llm_provider: 'ollama',
  total_ms: 1234,
  candidate_count: 5,
  kept_count: 3,
};

describe('TraceBrowser', () => {
  it('lists traces and opens a detail on row click', async () => {
    listTracesMock.mockResolvedValue({ traces: [SUMMARY] });
    getTraceMock.mockResolvedValue({
      trace_id: 't1',
      created_at: 0,
      question: 'What is DocRAG?',
      mode: 'stream',
      status: 'ok',
      config: null,
      candidates: [],
      prompt_messages: null,
      answer: 'A document Q&A system.',
      cited_source_numbers: [],
      timings: null,
      error: null,
      condensed_question: null,
    });

    render(<TraceBrowser onClose={vi.fn()} />);

    const row = await screen.findByTestId('trace-browser-row');
    expect(row).toHaveTextContent('What is DocRAG?');

    await userEvent.click(row);

    await waitFor(() => {
      expect(screen.getByTestId('trace-browser-detail')).toHaveTextContent('A document Q&A system.');
    });
    expect(getTraceMock).toHaveBeenCalledWith('t1');
  });

  it('shows an empty message when there are no traces', async () => {
    listTracesMock.mockResolvedValue({ traces: [] });
    render(<TraceBrowser onClose={vi.fn()} />);
    expect(await screen.findByText('No traces yet.')).toBeInTheDocument();
  });

  it('closes via the close button', async () => {
    listTracesMock.mockResolvedValue({ traces: [] });
    const onClose = vi.fn();
    render(<TraceBrowser onClose={onClose} />);
    await screen.findByText('No traces yet.');
    await userEvent.click(screen.getByTestId('trace-browser-close'));
    expect(onClose).toHaveBeenCalled();
  });
});
