import type { AnswerStatus, CitationResponse, TimingsResponse, UsageResponse } from './types';
import { ApiClientError } from './errors';

/** One frame of a streamed answer or summary (``/questions/stream``,
 *  ``/questions/summary/stream``). */
export type QuestionStreamEvent =
  | {
      type: 'stage';
      stage: string;
      step?: number;
      steps?: number;
      prompt_tokens?: number;
      eta_ms?: number;
    }
  | { type: 'sources'; sources: CitationResponse[]; trace_id: string | null }
  | { type: 'delta'; text: string }
  | {
      type: 'done';
      answer: string;
      sources: CitationResponse[];
      timings: TimingsResponse;
      trace_id: string | null;
      cached?: boolean;
      answer_status?: AnswerStatus;
      top_rerank_score?: number | null;
      weak_matches?: boolean;
      truncated?: boolean;
      citation_backfilled?: boolean;
      chunks_read?: number;
      chunks_total?: number;
      usage?: UsageResponse | null;
    }
  | { type: 'followups'; questions: string[] }
  | { type: 'error'; detail: string };

/** Read a ``text/event-stream`` response, calling ``onEvent`` per frame and
 *  ``onChunk`` per network chunk (the caller's idle timer). */
export async function readSseStream(
  response: Response,
  onEvent: (event: QuestionStreamEvent) => void,
  onChunk?: () => void,
): Promise<void> {
  if (!response.body) {
    throw new ApiClientError('Streaming response had no body.');
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  // Parsed by the SSE rules, not this server's exact framing: a proxy that rewrites
  // line endings to CRLF, splits one event's data over several `data:` lines or drops
  // the space after the colon used to make frames vanish — and a last frame with no
  // blank line after it (the connection closed) was dropped.
  let dataLines: string[] = [];
  const dispatch = () => {
    if (dataLines.length > 0) {
      onEvent(JSON.parse(dataLines.join('\n')) as QuestionStreamEvent);
    }
    dataLines = [];
  };
  const takeLine = (line: string) => {
    if (line === '') {
      dispatch();
      return;
    }
    if (line.startsWith(':')) return; // a comment: the keepalive ping
    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? '' : line.slice(colon + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'data') dataLines.push(value);
  };
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      onChunk?.();
      buffer += decoder.decode(value, { stream: true });
      // A lone "\r" at the very end may be the first half of a "\r\n": wait for more.
      const lines = buffer.split(/\r\n|\r(?!$)|\n/);
      buffer = lines.pop() ?? '';
      lines.forEach(takeLine);
    }
    buffer += decoder.decode();
    for (const line of buffer.split(/\r\n|\r|\n/)) takeLine(line);
    dispatch();
  } finally {
    // Leaving early — a backend `error` event, a malformed frame, an abort — used to
    // leave the body locked to this reader until GC. cancel() releases the lock and
    // tells the browser to stop pulling bytes we will never read.
    await reader.cancel().catch(() => {
      // Already errored or closed; nothing left to release.
    });
  }
}
