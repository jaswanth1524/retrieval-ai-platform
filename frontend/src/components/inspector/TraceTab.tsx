import { useState } from 'react';
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
      `${trace.candidates.filter((c) => c.kept).length} kept` +
        // Waiting for another question's rerank to finish: not this question's cost.
        (t.rerank_wait_ms ? ` · ${t.rerank_wait_ms.toFixed(0)}ms queued` : ''),
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
      trace.citation_retry_used
        ? 'no citations, retried'
        : trace.citation_backfilled
          ? `no citations, ${trace.cited_source_numbers.length} matched`
          : `${trace.cited_source_numbers.length} cited`,
    ],
    // Its own span since generate_ms stopped including it: the bars add up to the total.
    [
      'citation retry',
      t.citation_retry_ms ?? 0,
      'model',
      `${trace.cited_source_numbers.length} cited`,
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

/** The trace and request ids in full, to find this question in the server's logs. */
function TraceIds({ trace }: TraceTabProps) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    void navigator.clipboard?.writeText(trace.trace_id).then(
      () => setCopied(true),
      () => setCopied(false),
    );
  };
  return (
    <div className="trace-tab__ids mono" data-testid="trace-ids">
      <span>
        trace <span className="trace-tab__id">{trace.trace_id}</span>
      </span>
      {navigator.clipboard && (
        <button type="button" className="trace-tab__copy" onClick={copy} data-testid="trace-copy-id">
          {copied ? 'copied' : 'copy'}
        </button>
      )}
      {trace.request_id && (
        <span>
          request <span className="trace-tab__id">{trace.request_id}</span>
        </span>
      )}
      {trace.usage && (
        <span data-testid="trace-usage">
          tokens {trace.usage.prompt_tokens} in · {trace.usage.completion_tokens} out
        </span>
      )}
    </div>
  );
}

function TraceTab({ trace }: TraceTabProps) {
  const spans = buildSpans(trace);
  const total = trace.timings?.total_ms ?? 0;

  if (!trace.timings) {
    return (
      <div>
        <p className="inspector__empty" data-testid="trace-tab-empty">
          This trace has no timings — the request failed before the pipeline finished.
          {trace.error ? ` (${trace.error})` : ''}
        </p>
        <TraceIds trace={trace} />
      </div>
    );
  }

  return (
    <div className="trace-tab" data-testid="trace-tab">
      <p className="trace-tab__header mono">
        trace {trace.trace_id.slice(0, 6)} &middot; {(total / 1000).toFixed(2)}s &middot;{' '}
        {spans.length} spans
        {trace.timings.first_token_ms ? (
          <> &middot; first token {(trace.timings.first_token_ms / 1000).toFixed(2)}s</>
        ) : null}
      </p>
      <TraceIds trace={trace} />
      {trace.error && (
        // A stream that failed late has timings too: the error used to be hidden then.
        <p className="trace-tab__error" role="note" data-testid="trace-error">
          {trace.error}
        </p>
      )}

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

      {trace.query_variants && trace.query_variants.length > 0 && (
        <div className="trace-tab__note" data-testid="trace-variants">
          <p className="trace-tab__note-label mono">query variants</p>
          <ul className="trace-tab__variants">
            {trace.query_variants.map((variant) => (
              <li key={variant} className="mono">
                {variant}
              </li>
            ))}
          </ul>
        </div>
      )}

      {trace.condensed_question && (
        <div className="trace-tab__note">
          <p className="trace-tab__note-label mono">condensed query</p>
          <p className="trace-tab__note-body mono">{trace.condensed_question}</p>
        </div>
      )}

      {trace.weak_matches && (
        <div className="trace-tab__note" data-testid="trace-weak-matches">
          <p className="trace-tab__note-label mono">weak matches</p>
          <p className="trace-tab__note-body">
            No candidate cleared the rerank bar; the closest ones were sent to the model, marked as
            weak matches.
          </p>
        </div>
      )}

      {trace.condense_note && (
        <div className="trace-tab__note" data-testid="trace-condense-note">
          <p className="trace-tab__note-label mono">condense</p>
          <p className="trace-tab__note-body mono">{trace.condense_note}</p>
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
