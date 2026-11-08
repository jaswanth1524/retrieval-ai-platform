import type { ChatTurn } from '../ChatMessage';

interface AnswerFlagsProps {
  turn: ChatTurn;
  /** A best passage scoring under this (just over the bar) reads "weakly supported". */
  weakSupportBelow?: number;
}

/** The short flags under an answer: how it went (stopped, cached, not found, weak
 *  match, uncited, cut off…) and, for a summary, how much it read. */
export default function AnswerFlags({ turn, weakSupportBelow }: AnswerFlagsProps) {
  return (
    <>
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
        <span className="chat-message__flag" data-testid="chat-message-not-found">
          {turn.scoped ? 'not in the selected documents' : 'not in your documents'}
        </span>
      )}
      {turn.autoCited && (
        <span className="chat-message__flag chat-message__flag--warn" data-testid="chat-message-auto-cited">
          auto-cited
        </span>
      )}
      {turn.weakMatch && !turn.notFound && (
        <span
          className="chat-message__flag chat-message__flag--warn"
          title="No passage matched this question closely; the answer comes from the nearest ones — check it against its sources"
          data-testid="chat-message-weak-match"
        >
          weak match
        </span>
      )}
      {weakSupportBelow !== undefined &&
        !turn.weakMatch &&
        typeof turn.topRerankScore === 'number' &&
        turn.topRerankScore < weakSupportBelow &&
        !turn.notFound &&
        turn.sources.length > 0 && (
          <span
            className="chat-message__flag chat-message__flag--warn"
            title={`The best passage found scored ${turn.topRerankScore.toFixed(2)}, just over the bar for being used at all — the documents may only touch on this`}
            data-testid="chat-message-weak-support"
          >
            weakly supported
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
      {turn.coverage && turn.coverage.read < turn.coverage.total && (
        <span
          className="chat-message__flag"
          title="The document is longer than one summary reads (SUMMARY_MAX_BATCHES): it was read at evenly spaced passages"
          data-testid="chat-message-coverage"
        >
          read {turn.coverage.read} of {turn.coverage.total} passages
        </span>
      )}
      {turn.coverage?.skipped ? (
        <span
          className="chat-message__flag chat-message__flag--warn"
          title="The model failed twice on these parts of the document, so the summary was written without them"
          data-testid="chat-message-parts-skipped"
        >
          {turn.coverage.skipped} part{turn.coverage.skipped === 1 ? '' : 's'} left out
        </span>
      ) : null}
      {turn.coverage?.cutOff ? (
        <span
          className="chat-message__flag chat-message__flag--warn"
          title="Notes on these parts reached the output limit (LLM_MAX_TOKENS) and stopped; the summary may miss what came after"
          data-testid="chat-message-notes-cut"
        >
          notes cut off on {turn.coverage.cutOff} part{turn.coverage.cutOff === 1 ? '' : 's'}
        </span>
      ) : null}
      {turn.truncated && (
        <span
          className="chat-message__flag chat-message__flag--warn"
          title="The model reached its output limit (LLM_MAX_TOKENS) and stopped mid-answer"
          data-testid="chat-message-truncated"
        >
          cut off
        </span>
      )}
    </>
  );
}
