import { useEffect, useRef, useState } from 'react';
import type { CitationResponse } from '../api/types';
import ChatMessage, { type ChatTurn, type FeedbackPayload } from './ChatMessage';
import StreamingSkeleton from './StreamingSkeleton';
import './ChatThread.css';

interface ChatThreadProps {
  turns: ChatTurn[];
  pending: boolean;
  // Live label for the stage the in-flight question is on, from the server's `stage`
  // SSE frames (see useChat). Optional so a caller that doesn't track it keeps the old
  // hardcoded labels rather than rendering nothing.
  stage?: string | null;
  engineerMode: boolean;
  currentModelLabel?: string;
  onRetry?: (question: string) => void;
  // Regenerate (skips the answer cache); falls back to onRetry when absent.
  onRegenerate?: (question: string) => void;
  onEditQuestion?: (turnId: string, question: string) => void;
  feedbackEnabled?: boolean;
  onFeedback?: (payload: FeedbackPayload) => void;
  onOpenSource?: (filename: string, chunkId: string) => void;
  onCitationHover?: (citation: CitationResponse) => void;
  onCitationLeave?: () => void;
  /** Indexed documents available to ask about. The empty state told users to upload
   *  one even when the corpus already had some. Optional (0) for older callers. */
  documentCount?: number;
}

// How close to the bottom still counts as "following along". Wide enough to survive
// sub-pixel rounding and the last line's leading, narrow enough that a user who
// scrolled up to re-read is not yanked back down.
const PIN_THRESHOLD_PX = 48;

/** What the screen-reader status line says about the question in flight. */
function progressAnnouncement(
  pending: boolean,
  lastTurn: ChatTurn | undefined,
  stage: string | null | undefined,
  justFinished: boolean,
): string {
  if (pending) {
    if (lastTurn?.role === 'assistant' && lastTurn.content) return 'Answer streaming.';
    return stage ?? 'retrieving…';
  }
  // A stopped or cut-off answer is not "ready", and a failure is announced by its own
  // alert.
  if (
    justFinished &&
    lastTurn?.role === 'assistant' &&
    lastTurn.content &&
    !lastTurn.stopped &&
    !lastTurn.incomplete
  ) {
    return 'Answer ready.';
  }
  return '';
}

// The question a "Regenerate" click on an assistant turn should re-ask. Read from the
// nearest preceding user turn by array position rather than stored on the assistant
// turn itself — turns are strictly alternating in the normal flow, but searching
// backward is also correct if an error turn ever lands in between, and it avoids
// growing the persisted localStorage v2 chat-history schema for a value that's always
// reconstructible from position.
// One pass over the thread: searching backward from every assistant turn made each
// render quadratic in conversation length.
function precedingUserQuestions(turns: ChatTurn[]): (string | undefined)[] {
  let lastQuestion: string | undefined;
  return turns.map((turn) => {
    const preceding = lastQuestion;
    if (turn.role === 'user') lastQuestion = turn.content;
    return preceding;
  });
}

function ChatThread({
  turns,
  pending,
  stage,
  engineerMode,
  currentModelLabel,
  onRetry,
  onRegenerate,
  onEditQuestion,
  feedbackEnabled,
  onFeedback,
  onOpenSource,
  onCitationHover,
  onCitationLeave,
  documentCount = 0,
}: ChatThreadProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  // Whether the view should keep tracking the bottom. A ref, not state, so a scroll
  // event never costs a render.
  const pinnedRef = useRef(true);
  const turnCountRef = useRef(turns.length);
  // Where the effect below last parked the scroll, so its own scroll event can be told
  // apart from the user's. Without this the component unpins itself: the effect scrolls
  // against the height it can see, the next delta grows the content, and the queued
  // scroll event then measures a gap against the *new* height and concludes the user
  // scrolled away — observed in a browser as scrollTop frozen while scrollHeight climbed.
  // Position, not a timer: deltas can arrive several times per animation frame, so a
  // frame-based guard would swallow genuine user scrolls during a fast stream.
  const autoScrollTopRef = useRef(-1);
  // Set when a question this thread watched finishes, so "Answer ready." is said once
  // for a new answer — never for history shown on a reload or conversation switch.
  const [justFinished, setJustFinished] = useState(false);
  const wasPendingRef = useRef(pending);
  useEffect(() => {
    if (pending) setJustFinished(false);
    else if (wasPendingRef.current) setJustFinished(true);
    wasPendingRef.current = pending;
  }, [pending]);
  const firstTurnId = turns[0]?.id;
  useEffect(() => {
    setJustFinished(false);
  }, [firstTurnId]);

  const handleScroll = () => {
    const element = containerRef.current;
    if (!element) return;
    if (Math.abs(element.scrollTop - autoScrollTopRef.current) <= 1) return;
    pinnedRef.current =
      element.scrollHeight - element.scrollTop - element.clientHeight <= PIN_THRESHOLD_PX;
  };

  // Depends on `turns` itself, not `turns.length`: useChat rebuilds the turn array on
  // every streamed delta, so this runs as the answer grows. The previous
  // [turns.length, pending] deps changed only when a turn was added or the request
  // settled — neither happens mid-stream — so a long answer grew straight past the
  // bottom of the viewport with no autoscroll until the next question.
  useEffect(() => {
    const isNewTurn = turns.length !== turnCountRef.current;
    turnCountRef.current = turns.length;
    // Asking something new always pulls the view back down; mid-answer growth only
    // does so while the user is still at the bottom following along.
    if (isNewTurn) pinnedRef.current = true;
    if (!pinnedRef.current) return;
    const element = containerRef.current;
    if (!element) return;
    // Jump rather than animate: a smooth scroll per delta queues dozens of competing
    // animations and visibly stutters.
    element.scrollTop = element.scrollHeight;
    // Read back rather than reusing scrollHeight: the browser clamps the assignment to
    // scrollHeight - clientHeight, and the comparison in handleScroll needs the value
    // the element actually holds.
    autoScrollTopRef.current = element.scrollTop;
  }, [turns, pending]);

  const lastTurn = turns[turns.length - 1];
  // One polite status line for the whole thread, always mounted so the first change is
  // announced too. The answer text itself is not a live region: it changes on every
  // streamed token, and a polite region over it re-reads the whole answer each time.
  // One per message used to leave dozens of regions in a long conversation.
  const statusRegion = (
    <span className="chat-thread__status" role="status" data-testid="chat-status">
      {progressAnnouncement(pending, lastTurn, stage, justFinished)}
    </span>
  );

  if (turns.length === 0 && !pending) {
    return (
      <div className="chat-thread chat-thread--empty">
        {statusRegion}
        <p>
          {documentCount > 0
            ? `Ask a question about your ${documentCount === 1 ? 'document' : `${documentCount} documents`}.`
            : 'Upload a document, then ask a question about it.'}
        </p>
      </div>
    );
  }

  // The assistant turn is created lazily on the first `sources`/`delta` event (see
  // useChat), so while pending is true but the last turn is still the user's question,
  // there's no ChatTurn yet to attach a skeleton to — render a placeholder in its slot.
  const awaitingFirstEvent = pending && lastTurn?.role !== 'assistant';
  const questions = precedingUserQuestions(turns);

  return (
    <div
      className="chat-thread"
      ref={containerRef}
      onScroll={handleScroll}
      data-testid="chat-thread"
    >
      {statusRegion}
      <div className="chat-thread__rail">
        {turns.map((turn, index) => (
          <ChatMessage
            key={turn.id}
            turn={turn}
            engineerMode={engineerMode}
            currentModelLabel={currentModelLabel}
            streamStage={
              pending && turn.id === lastTurn?.id && turn.role === 'assistant'
                ? (stage ?? 'generating answer…')
                : undefined
            }
            onRetry={onRetry}
            onRegenerate={onRegenerate}
            onEditQuestion={onEditQuestion}
            regenerateQuestion={turn.role === 'assistant' ? questions[index] : undefined}
            busy={pending}
            feedbackEnabled={feedbackEnabled}
            onFeedback={onFeedback}
            onOpenSource={onOpenSource}
            onCitationHover={onCitationHover}
            onCitationLeave={onCitationLeave}
          />
        ))}
        {awaitingFirstEvent && (
          <div className="chat-message" data-testid="chat-pending">
            <div className="chat-message__row">
              <span className="chat-message__gutter chat-message__gutter--answer" aria-hidden="true">
                A
              </span>
              <div className="chat-message__answer">
                <StreamingSkeleton stage={stage ?? 'retrieving…'} />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ChatThread;
