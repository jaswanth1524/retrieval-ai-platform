import { useEffect, useState } from 'react';
import { ApiClientError, api } from '../api/client';
import type { TraceDetailResponse } from '../api/types';

/** What the inspector knows about one turn's trace.
 *
 * `unavailable` is deliberately separate from `error`: a trace can be legitimately
 * absent — the operator set TRACE_ENABLED=false, or the server's ring buffer
 * (TRACE_MAX_RETAINED, 100 by default) evicted it, or the process restarted. None of
 * those are faults, and rendering them as failures would make a healthy server look
 * broken every time an older turn is inspected.
 */
export type TraceState =
  | { status: 'loading' }
  | { status: 'ready'; trace: TraceDetailResponse }
  | { status: 'unavailable'; reason: string }
  | { status: 'error'; message: string }
  | { status: 'pending'; reason: string };

const NO_TRACE_ID =
  'No trace was recorded for this turn. Tracing may be off on the server (TRACE_ENABLED).';
const EVICTED =
  'This trace is no longer on the server. Only the most recent traces are kept, and they are lost on restart.';
const STILL_GENERATING = 'The trace will be available once this answer finishes generating.';

// Traces are immutable once written, so a fetched one can be reused for the life of the
// page. Capped because a long session inspects many turns and each trace carries the
// full prompt plus every candidate — tens of KB apiece.
const MAX_CACHED = 40;
const cache = new Map<string, TraceDetailResponse>();

function remember(traceId: string, trace: TraceDetailResponse): void {
  cache.set(traceId, trace);
  while (cache.size > MAX_CACHED) {
    const oldest = cache.keys().next();
    if (oldest.done) break;
    cache.delete(oldest.value);
  }
}

/**
 * Fetch a turn's trace once and cache it. Null id means the turn has no trace.
 *
 * ``ready`` must be false while the trace's own turn is still streaming. The server
 * sends ``trace_id`` in the SSE ``sources`` event — before generation even starts, so
 * the UI can still link a mid-stream failure to a trace — but only writes the trace
 * itself into the server's TraceStore at the terminal event (``done`` or an error),
 * which can be many seconds later. A fetch issued in that gap reliably 404s, and
 * because ``traceId`` itself never changes between ``sources`` and ``done``, the
 * effect below would never re-run to retry — permanently showing "no longer on the
 * server" for a trace that exists moments later. Found via real Ollama generation
 * (~19s of gap); a mocked SSE stream that resolves sources/delta/done synchronously
 * can never reproduce this race.
 */
export function useTrace(traceId: string | null, ready = true): TraceState {
  // Keyed by the trace it describes: when the inspected turn changes, the state from
  // the previous one would otherwise paint for a frame before the effect below reset it.
  const [keyed, setKeyed] = useState<{ id: string | null; state: TraceState }>(() => ({
    id: traceId,
    state: initialState(traceId, ready),
  }));
  const setState = (state: TraceState) => setKeyed({ id: traceId, state });

  useEffect(() => {
    if (traceId === null) {
      setState({ status: 'unavailable', reason: NO_TRACE_ID });
      return;
    }
    const hit = cache.get(traceId);
    if (hit) {
      setState({ status: 'ready', trace: hit });
      return;
    }
    if (!ready) {
      // Deliberately no fetch here — issuing one now is the exact race described
      // above. Once `ready` flips true this effect re-runs (it's a dependency) and
      // fetches for real.
      setState({ status: 'pending', reason: STILL_GENERATING });
      return;
    }

    // Abort on unmount or on a switch to another trace, so a slow response for the
    // previously selected turn can never overwrite the current one.
    const controller = new AbortController();
    setState({ status: 'loading' });
    api
      .getTrace(traceId, controller.signal)
      .then((trace) => {
        remember(traceId, trace);
        if (!controller.signal.aborted) setState({ status: 'ready', trace });
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        if (err instanceof ApiClientError && err.statusCode === 404) {
          setState({ status: 'unavailable', reason: EVICTED });
          return;
        }
        setState({
          status: 'error',
          message: err instanceof ApiClientError ? err.message : 'Could not load the trace.',
        });
      });

    return () => controller.abort();
    // setState only closes over traceId, which is already a dependency.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [traceId, ready]);

  return keyed.id === traceId ? keyed.state : initialState(traceId, ready);
}

function initialState(traceId: string | null, ready: boolean): TraceState {
  if (traceId === null) return { status: 'unavailable', reason: NO_TRACE_ID };
  const cached = cache.get(traceId);
  if (cached) return { status: 'ready', trace: cached };
  return ready ? { status: 'loading' } : { status: 'pending', reason: STILL_GENERATING };
}

/** Test seam: drop the module-level cache so cases don't leak into one another. */
export function clearTraceCache(): void {
  cache.clear();
}
