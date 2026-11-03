import { useState } from 'react';
import type { ChatTurn } from '../components/ChatMessage';

/** Which turn the inspector describes, and whether its trace can be fetched yet. */
export function useInspectorTarget(turns: ChatTurn[], streamingHere: boolean) {
  // null follows the newest assistant turn, so the panel keeps up with the conversation
  // on its own; clicking a trace row pins it to that specific turn instead.
  const [pinnedTraceId, setPinnedTraceId] = useState<string | null>(null);

  // The turn the inspector describes: the one pinned by a trace-row click, else the
  // most recent assistant turn so the panel tracks the conversation without a click.
  // A pin this conversation doesn't hold describes no turn here: falling back to the
  // latest answer showed that answer's trace instead of the row that was clicked.
  const inspectedTurn = pinnedTraceId
    ? turns.find((turn) => turn.traceId === pinnedTraceId)
    : [...turns].reverse().find((turn) => turn.role === 'assistant');
  // A pinned trace from the trace browser may belong to a turn this conversation never
  // held (another session, or one since cleared) — fall back to the id itself so the
  // Retrieval and Trace tabs can still fetch and render it.
  const inspectedTraceId = inspectedTurn?.traceId ?? pinnedTraceId;
  // The server sends trace_id in the `sources` SSE event, before generation even
  // starts (so a mid-stream failure can still be linked to a trace) — but only writes
  // the trace itself once the turn reaches its terminal event, which is also the
  // moment `timings` gets set (see useChat's onDone handler). Fetching in that gap
  // 404s; a trace pinned from the trace browser is never mid-stream, so it defaults
  // ready. Found via a real ~19s Ollama generation — a mocked instant SSE stream can't
  // reproduce the gap this guards against.
  // Only the in-flight turn can be in that gap. A turn that ended without `done`
  // (cancelled, or failed mid-stream) never gets timings, but the server still wrote an
  // error trace for it — gating on timings alone left its Inspector stuck forever.
  const inspectedTurnInFlight =
    streamingHere && inspectedTurn !== undefined && inspectedTurn.id === turns[turns.length - 1]?.id;
  const inspectedTraceReady = inspectedTurn
    ? inspectedTurn.timings !== null || !inspectedTurnInFlight
    : true;

  return { pinnedTraceId, setPinnedTraceId, inspectedTurn, inspectedTraceId, inspectedTraceReady };
}
