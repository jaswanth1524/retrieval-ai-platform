import { useEffect, useRef } from 'react';
import type { CitationResponse } from '../api/types';
import ChatMessage, { type ChatTurn } from './ChatMessage';
import StreamingSkeleton from './StreamingSkeleton';
import './ChatThread.css';

interface ChatThreadProps {
  turns: ChatTurn[];
  pending: boolean;
  engineerMode: boolean;
  currentModelLabel?: string;
  onRetry?: (question: string) => void;
  onOpenSource?: (filename: string, chunkId: string) => void;
  onCitationHover?: (citation: CitationResponse) => void;
  onCitationLeave?: () => void;
}

// How close to the bottom still counts as "following along". Wide enough to survive
// sub-pixel rounding and the last line's leading, narrow enough that a user who
// scrolled up to re-read is not yanked back down.
const PIN_THRESHOLD_PX = 48;

function ChatThread({
  turns,
  pending,
  engineerMode,
  currentModelLabel,
  onRetry,
  onOpenSource,
  onCitationHover,
  onCitationLeave,
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

  if (turns.length === 0 && !pending) {
    return (
      <div className="chat-thread chat-thread--empty">
        <p>Upload a document, then ask a question about it.</p>
      </div>
    );
  }

  const lastTurn = turns[turns.length - 1];
  // The assistant turn is created lazily on the first `sources`/`delta` event (see
  // useChat), so while pending is true but the last turn is still the user's question,
  // there's no ChatTurn yet to attach a skeleton to — render a placeholder in its slot.
  const awaitingFirstEvent = pending && lastTurn?.role !== 'assistant';

  return (
    <div
      className="chat-thread"
      ref={containerRef}
      onScroll={handleScroll}
      data-testid="chat-thread"
    >
      <div className="chat-thread__rail">
        {turns.map((turn) => (
          <ChatMessage
            key={turn.id}
            turn={turn}
            engineerMode={engineerMode}
            currentModelLabel={currentModelLabel}
            streamStage={pending && turn.id === lastTurn?.id && turn.role === 'assistant' ? 'generating answer…' : undefined}
            onRetry={onRetry}
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
                <StreamingSkeleton stage="retrieving…" />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ChatThread;
