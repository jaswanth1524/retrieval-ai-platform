import type { TraceDetailResponse } from '../../api/types';

interface TraceTabProps {
  trace: TraceDetailResponse;
}

type SpanKind = 'io' | 'model' | 'root';

interface Span {
  name: string;
  ms: number;
  startMs: number;
  kind: SpanKind;
  attribute: string;
}

/** Lay the stages out end to end in real pipeline order.
 *
 * The API reports per-stage durations, not absolute timestamps, so offsets are the
 * running sum — which is faithful because the pipeline is strictly sequential. Order
 * matters and is not the order of the fields: query expansion runs BEFORE embedding
 * (it rewrites the query) and context expansion AFTER rerank (it widens the winners).
 * `expand_ms` used to conflate the two, which would have put a span in the wrong place.
 */
function buildSpans(trace: TraceDetailResponse): Span[] {
  const t = trace.timings;
  if (!t) return [];

  const stages: Array<[string, number, SpanKind, string]> = [
    ['condense', t.condense_ms, 'model', `${trace.history_message_count ?? 0} history messages`],
    [
      'query expansion',
      t.query_expansion_ms ?? 0,
      'model',
      `${trace.query_variants?.length ?? 0} variants`,
    ],
    ['embed', t.embed_ms, 'io', 'dense + sparse'],
    ['search', t.search_ms, 'io', `${trace.candidates.length} fused candidates`],
    [
      'rerank',
      t.rerank_ms,
      'model',
      `${trace.candidates.filter((c) => c.kept).length} kept`,
    ],
    [
      'context expansion',
      t.context_expansion_ms ?? 0,
      'io',
      `${trace.candidates.filter((c) => c.neighbor_expanded).length} expanded`,
    ],
    [
      'generate',
      t.generate_ms,
      'model',
      trace.citation_retry_used ? 'included a citation retry' : `${trace.cited_source_numbers.length} cited`,
    ],
  ];

  const spans: Span[] = [];
  let cursor = 0;
  for (const [name, ms, kind, attribute] of stages) {
    // Skip stages that did not run — a zero-width bar is noise, not information.
    if (ms <= 0) continue;
    spans.push({ name, ms, startMs: cursor, kind, attribute });
    cursor += ms;
  }
  return spans;
}

function TraceTab({ trace }: TraceTabProps) {
  const spans = buildSpans(trace);
  const total = trace.timings?.total_ms ?? 0;

  if (!trace.timings) {
    return (
      <p className="inspector__empty" data-testid="trace-tab-empty">
        This trace has no timings — the request failed before the pipeline finished.
        {trace.error ? ` (${trace.error})` : ''}
      </p>
    );
  }

  return (
    <div className="trace-tab" data-testid="trace-tab">
      <p className="trace-tab__header mono">
        trace {trace.trace_id.slice(0, 6)} &middot; {(total / 1000).toFixed(2)}s &middot;{' '}
        {spans.length} spans
      </p>

      <div className="trace-tab__span trace-tab__span--root">
        <div className="trace-tab__span-head mono">
          <span>total</span>
          <span>{total.toFixed(0)}ms</span>
        </div>
        <div className="trace-tab__track">
          <span className="trace-tab__bar trace-tab__bar--root" style={{ left: '0%', width: '100%' }} />
        </div>
        <p className="trace-tab__attr mono">
          {trace.mode} &middot; {trace.status}
        </p>
      </div>

      {spans.map((span) => (
        <div key={span.name} className="trace-tab__span" data-testid="trace-span">
          <div className="trace-tab__span-head mono">
            <span>{span.name}</span>
            <span>{span.ms.toFixed(0)}ms</span>
          </div>
          <div className="trace-tab__track">
            <span
              className={`trace-tab__bar trace-tab__bar--${span.kind}`}
              style={{
                left: total > 0 ? `${(span.startMs / total) * 100}%` : '0%',
                // Floor the width so a sub-millisecond stage is still visible.
                width: total > 0 ? `max(2px, ${(span.ms / total) * 100}%)` : '2px',
              }}
            />
          </div>
          <p className="trace-tab__attr mono">{span.attribute}</p>
        </div>
      ))}

      {trace.condensed_question && (
        <div className="trace-tab__note">
          <p className="trace-tab__note-label mono">condensed query</p>
          <p className="trace-tab__note-body mono">{trace.condensed_question}</p>
        </div>
      )}

      {trace.prompt_messages && (
        <details className="trace-tab__prompt">
          <summary className="mono">
            prompt sent ({trace.prompt_messages.length} messages)
            {trace.citation_retry_used ? ' · after citation retry' : ''}
          </summary>
          {trace.prompt_messages.map((message, index) => (
            <div key={index} className="trace-tab__message">
              <p className="trace-tab__message-role mono">{message.role}</p>
              <pre className="trace-tab__message-body">{message.content}</pre>
            </div>
          ))}
        </details>
      )}
    </div>
  );
}

export default TraceTab;
