import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import TracesPanel from '../../src/components/TracesPanel';
import type { TraceSummaryResponse } from '../../src/api/types';

function makeTrace(overrides: Partial<TraceSummaryResponse> = {}): TraceSummaryResponse {
  return {
    trace_id: 't1',
    created_at: Date.now() / 1000,
    question: 'liability cap acme msa',
    mode: 'sync',
    status: 'ok',
    llm_provider: 'ollama',
    total_ms: 2310,
    candidate_count: 24,
    kept_count: 8,
    ...overrides,
  };
}

describe('TracesPanel', () => {
  it('shows a loading message while traces is null', () => {
    render(<TracesPanel traces={null} error={null} onSelect={vi.fn()} />);
    expect(screen.getByText('Loading…')).toBeInTheDocument();
  });

  it('shows an empty state when there are no traces', () => {
    render(<TracesPanel traces={[]} error={null} onSelect={vi.fn()} />);
    expect(screen.getByText('No traces yet.')).toBeInTheDocument();
  });

  it('shows the error message when loading fails', () => {
    render(<TracesPanel traces={null} error="Could not load traces." onSelect={vi.fn()} />);
    expect(screen.getByRole('alert')).toHaveTextContent('Could not load traces.');
  });

  it('renders a row per trace with latency colored by threshold', () => {
    const traces = [
      makeTrace({ trace_id: 'fast', total_ms: 1740 }),
      makeTrace({ trace_id: 'warn', total_ms: 4080 }),
      makeTrace({ trace_id: 'slow', total_ms: 9120 }),
    ];
    render(<TracesPanel traces={traces} error={null} onSelect={vi.fn()} />);

    const rows = screen.getAllByTestId('traces-panel-row');
    expect(rows).toHaveLength(3);
    expect(rows[0].querySelector('.traces-panel__latency--good')).not.toBeNull();
    expect(rows[1].querySelector('.traces-panel__latency--warn')).not.toBeNull();
    expect(rows[2].querySelector('.traces-panel__latency--bad')).not.toBeNull();
  });

  it('fires onSelect with the trace id when a row is clicked', async () => {
    const onSelect = vi.fn();
    render(<TracesPanel traces={[makeTrace({ trace_id: 'abc123' })]} error={null} onSelect={onSelect} />);

    await userEvent.click(screen.getByTestId('traces-panel-row'));
    expect(onSelect).toHaveBeenCalledWith('abc123');
  });
});
