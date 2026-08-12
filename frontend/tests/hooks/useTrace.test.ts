import { renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiClientError } from '../../src/api/client';
import { clearTraceCache, useTrace } from '../../src/hooks/useTrace';
import type { TraceDetailResponse } from '../../src/api/types';

const getTrace = vi.fn();
vi.mock('../../src/api/client', async () => {
  const actual = await vi.importActual<typeof import('../../src/api/client')>(
    '../../src/api/client',
  );
  return { ...actual, api: { getTrace: (...args: unknown[]) => getTrace(...args) } };
});

function makeTrace(overrides: Partial<TraceDetailResponse> = {}): TraceDetailResponse {
  return {
    trace_id: 'trace-1',
    created_at: 0,
    question: 'q',
    mode: 'stream',
    status: 'ok',
    config: null,
    candidates: [],
    prompt_messages: null,
    answer: 'a',
    cited_source_numbers: [],
    timings: null,
    error: null,
    ...overrides,
  };
}

describe('useTrace', () => {
  beforeEach(() => {
    getTrace.mockReset();
    clearTraceCache();
  });

  it('reports a turn with no trace id as unavailable, not an error', () => {
    const { result } = renderHook(() => useTrace(null));

    expect(result.current.status).toBe('unavailable');
    expect(getTrace).not.toHaveBeenCalled();
  });

  it('loads a trace and exposes it', async () => {
    getTrace.mockResolvedValue(makeTrace());

    const { result } = renderHook(() => useTrace('trace-1'));

    await waitFor(() => expect(result.current.status).toBe('ready'));
    expect(result.current).toMatchObject({ trace: { trace_id: 'trace-1' } });
  });

  it('treats a 404 as unavailable — an evicted trace is not a failure', async () => {
    // TRACE_MAX_RETAINED is 100 and traces are lost on restart, so inspecting an older
    // turn legitimately 404s. Rendering that as an error makes a healthy server look
    // broken.
    getTrace.mockRejectedValue(new ApiClientError('No trace found with id x.', 404));

    const { result } = renderHook(() => useTrace('gone'));

    await waitFor(() => expect(result.current.status).toBe('unavailable'));
    expect(result.current).toMatchObject({ reason: expect.stringContaining('no longer') });
  });

  it('surfaces a genuine failure as an error', async () => {
    getTrace.mockRejectedValue(new ApiClientError('API returned HTTP 500.', 500));

    const { result } = renderHook(() => useTrace('boom'));

    await waitFor(() => expect(result.current.status).toBe('error'));
  });

  it('fetches a given trace only once across remounts', async () => {
    getTrace.mockResolvedValue(makeTrace());

    const first = renderHook(() => useTrace('trace-1'));
    await waitFor(() => expect(first.result.current.status).toBe('ready'));
    first.unmount();

    const second = renderHook(() => useTrace('trace-1'));

    // Served from cache synchronously — no second request, no loading flash.
    expect(second.result.current.status).toBe('ready');
    expect(getTrace).toHaveBeenCalledTimes(1);
  });

  it('aborts the in-flight request when the trace id changes', async () => {
    getTrace.mockImplementation(
      (_id: string, signal: AbortSignal) =>
        new Promise((_resolve, reject) => {
          signal.addEventListener('abort', () => reject(new Error('aborted')));
        }),
    );

    const { rerender } = renderHook(({ id }) => useTrace(id), {
      initialProps: { id: 'first' },
    });
    const firstSignal = getTrace.mock.calls[0][1] as AbortSignal;
    expect(firstSignal.aborted).toBe(false);

    rerender({ id: 'second' });

    // Otherwise a slow response for the previously selected turn could land after the
    // switch and overwrite the current one.
    expect(firstSignal.aborted).toBe(true);
  });

  describe('ready gate', () => {
    // Regression coverage for a real bug found via live Ollama generation (~19s
    // between the `sources` SSE event exposing trace_id and the server actually
    // writing the trace at the `done` event). A mocked SSE stream that resolves
    // sources/delta/done in the same tick cannot reproduce this gap — every other
    // test in this file effectively fetches "at done time" already.

    it('does not fetch while not ready, even though a traceId is already known', () => {
      const { result } = renderHook(() => useTrace('trace-1', false));

      expect(result.current.status).toBe('pending');
      expect(getTrace).not.toHaveBeenCalled();
    });

    it('fetches once ready flips true, using the SAME unchanged traceId', async () => {
      // This is the crux of the bug: traceId never changes between `sources` and
      // `done`, so only `ready` toggling can be what triggers the (re)fetch.
      getTrace.mockResolvedValue(makeTrace());
      const { result, rerender } = renderHook(({ ready }) => useTrace('trace-1', ready), {
        initialProps: { ready: false },
      });
      expect(result.current.status).toBe('pending');
      expect(getTrace).not.toHaveBeenCalled();

      rerender({ ready: true });

      await waitFor(() => expect(result.current.status).toBe('ready'));
      expect(getTrace).toHaveBeenCalledTimes(1);
    });

    it('a 404 while not-ready must never be possible: no request means no false eviction verdict', () => {
      // Guards the actual failure mode: without the ready gate, a fetch issued during
      // the gap 404s and — because traceId is unchanged — never retries, permanently
      // showing "no longer on the server" for a trace that exists moments later.
      getTrace.mockRejectedValue(new ApiClientError('No trace found with id trace-1.', 404));

      const { result } = renderHook(() => useTrace('trace-1', false));

      expect(result.current.status).toBe('pending');
      expect(getTrace).not.toHaveBeenCalled();
    });
  });
});
