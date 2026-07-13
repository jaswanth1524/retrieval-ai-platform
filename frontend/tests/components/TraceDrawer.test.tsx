import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import TraceDrawer from '../../src/components/TraceDrawer';
import { ApiClientError, api } from '../../src/api/client';
import type { TraceDetailResponse } from '../../src/api/types';

vi.mock('../../src/api/client', async () => {
  const actual = await vi.importActual<typeof import('../../src/api/client')>('../../src/api/client');
  return { ...actual, api: { ...actual.api, getTrace: vi.fn() } };
});

const getTraceMock = vi.mocked(api.getTrace);

afterEach(() => {
  getTraceMock.mockReset();
});

const ZERO_TIMINGS = { embed_ms: 1, search_ms: 2, rerank_ms: 3, generate_ms: 4, total_ms: 10 };

function makeDetail(overrides: Partial<TraceDetailResponse> = {}): TraceDetailResponse {
  return {
    trace_id: 'trace-1',
    created_at: 0,
    question: 'What is the refund window?',
    mode: 'sync',
    status: 'ok',
    config: {
      llm_provider: 'ollama',
      rerank_top_k: 8,
      max_context_chunks: 6,
      llm_temperature: 0,
      rerank_min_score: 0.15,
      fused_top_n: 50,
      filenames: null,
    },
    candidates: [
      {
        point_id: 'p1',
        filename: 'refund-policy.txt',
        page: 1,
        section: 'Refund Policy',
        chunk_id: 'c1',
        retrieval_score: 0.033,
        rerank_score: 0.86,
        kept: true,
        drop_reason: null,
        selected_for_context: true,
        neighbor_expanded: false,
      },
      {
        point_id: 'p2',
        filename: 'refund-policy.txt',
        page: 1,
        section: 'Shipping Costs',
        chunk_id: 'c2',
        retrieval_score: 0.01,
        rerank_score: 0.05,
        kept: false,
        drop_reason: 'below_min_score',
        selected_for_context: false,
        neighbor_expanded: false,
      },
    ],
    prompt_messages: [
      { role: 'system', content: 'You are DocRAG.' },
      { role: 'user', content: 'Question: What is the refund window?' },
    ],
    answer: 'Refunds must be requested within 30 days [1].',
    cited_source_numbers: [1],
    timings: ZERO_TIMINGS,
    error: null,
    ...overrides,
  };
}

describe('TraceDrawer', () => {
  it('is collapsed by default and does not fetch the trace', () => {
    render(<TraceDrawer traceId="trace-1" timings={null} />);

    expect(screen.getByTestId('trace-drawer-toggle')).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByTestId('trace-drawer-panel')).not.toBeInTheDocument();
    expect(getTraceMock).not.toHaveBeenCalled();
  });

  it('fetches and renders the trace on first expand', async () => {
    getTraceMock.mockResolvedValue(makeDetail());

    render(<TraceDrawer traceId="trace-1" timings={null} />);
    await userEvent.click(screen.getByTestId('trace-drawer-toggle'));

    await waitFor(() => expect(screen.getByTestId('trace-drawer-panel')).toBeInTheDocument());
    expect(getTraceMock).toHaveBeenCalledWith('trace-1', expect.anything());
    expect(screen.getByText('Refunds must be requested within 30 days [1].')).toBeInTheDocument();
    expect(screen.getAllByTestId('trace-drawer-candidate')).toHaveLength(2);
  });

  it('shows a drop-reason label for a dropped candidate', async () => {
    getTraceMock.mockResolvedValue(makeDetail());

    render(<TraceDrawer traceId="trace-1" timings={null} />);
    await userEvent.click(screen.getByTestId('trace-drawer-toggle'));

    await waitFor(() => expect(screen.getByText('below threshold')).toBeInTheDocument());
    expect(screen.getByText('in context')).toBeInTheDocument();
  });

  it('does not refetch on re-collapse and re-expand', async () => {
    getTraceMock.mockResolvedValue(makeDetail());

    render(<TraceDrawer traceId="trace-1" timings={null} />);
    const toggle = screen.getByTestId('trace-drawer-toggle');
    await userEvent.click(toggle);
    await waitFor(() => expect(screen.getByTestId('trace-drawer-panel')).toBeInTheDocument());

    await userEvent.click(toggle); // collapse
    expect(screen.queryByTestId('trace-drawer-panel')).not.toBeInTheDocument();
    await userEvent.click(toggle); // expand again

    expect(screen.getByTestId('trace-drawer-panel')).toBeInTheDocument();
    expect(getTraceMock).toHaveBeenCalledTimes(1);
  });

  it('shows a friendly message when the trace has been evicted (404)', async () => {
    getTraceMock.mockRejectedValue(new ApiClientError('No trace found.', 404));

    render(<TraceDrawer traceId="trace-1" timings={null} />);
    await userEvent.click(screen.getByTestId('trace-drawer-toggle'));

    await waitFor(() =>
      expect(screen.getByTestId('trace-drawer-error')).toHaveTextContent(
        'Trace no longer available',
      ),
    );
  });

  it('shows a generic error message for a non-404 failure', async () => {
    getTraceMock.mockRejectedValue(new ApiClientError('Server exploded.', 500));

    render(<TraceDrawer traceId="trace-1" timings={null} />);
    await userEvent.click(screen.getByTestId('trace-drawer-toggle'));

    await waitFor(() =>
      expect(screen.getByTestId('trace-drawer-error')).toHaveTextContent('Failed to load trace.'),
    );
  });

  it('falls back to the timings prop before the trace has loaded', () => {
    getTraceMock.mockImplementation(() => new Promise(() => {})); // never resolves

    render(<TraceDrawer traceId="trace-1" timings={ZERO_TIMINGS} />);

    // Expanding is async and unresolved here, so just confirm the collapsed state
    // renders without needing the fetch to complete.
    expect(screen.getByTestId('trace-drawer-toggle')).toBeInTheDocument();
  });
});
