import { useEffect, useRef, useState } from 'react';
import { MAX_QUESTION_CHARS } from '../../utils/limits';
import { isImeComposing } from '../../utils/keyboard';
import { enterAddsNewline } from '../../utils/platform';
import type { ChatTurn } from '../ChatMessage';
import { type AskAgain, askAgain } from './askAgain';

interface QuestionTurnProps {
  turn: ChatTurn;
  /** An answer is streaming: sending (an edit, Ask again) waits. */
  busy: boolean;
  /** No answer followed it: "Ask again" is offered. */
  unanswered: boolean;
  onRetry?: AskAgain;
  /** Ask the edited question in a new conversation. */
  onEditQuestion?: (turnId: string, question: string) => void;
}

/** A question in the thread: its text, the ✎ editor, and "Ask again" when it got no
 *  answer. */
export default function QuestionTurn({ turn, busy, unanswered, onRetry, onEditQuestion }: QuestionTurnProps) {
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
    // While an answer streams the edit can't be sent (Send is disabled): Enter used to
    // close the editor and drop the edit, since the resend then refused silently.
    if (!question || !onEditQuestion || busy) return;
    setEditing(false);
    onEditQuestion(turn.id, question);
  };

  return (
    <>
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
                } else if (event.key === 'Enter' && !event.shiftKey && !enterAddsNewline()) {
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
            {busy && (
              <p className="chat-message__edit-hint" data-testid="chat-message-edit-busy">
                Wait for the current answer to finish, then ask.
              </p>
            )}
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
            <p className="chat-message__question" dir="auto">{turn.content}</p>
            {/* A summary's text is a label, not the request: editing it would send a
                plain question, so summaries are asked again from the corpus panel. */}
            {onEditQuestion && !turn.task && (
              <button
                ref={editButtonRef}
                type="button"
                className="chat-message__edit-button"
                onClick={() => {
                  if (busy) return;
                  setDraft(turn.content);
                  setEditing(true);
                }}
                aria-disabled={busy || undefined}
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
      {unanswered && onRetry && !editing && (
        <div className="chat-message__unanswered" data-testid="chat-message-unanswered">
          <span>No answer to this question.</span>
          <button
            type="button"
            className="chat-message__action"
            onClick={() => !busy && askAgain(onRetry, turn.content, turn.task, turn.narrowing)}
            aria-disabled={busy || undefined}
            data-testid="chat-message-ask-again"
          >
            Ask again
          </button>
        </div>
      )}
    </>
  );
}
