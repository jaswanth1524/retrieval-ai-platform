import { memo, useEffect, useRef, useState } from 'react';
import ConfirmInline from './ConfirmInline';
import {
  basketToMarkdown,
  quotesToCsv,
  quotesToMarkdown,
  type BasketPassage,
  type Quote,
} from '../utils/collected';
import { copyText } from '../utils/clipboard';
import { downloadFile } from '../utils/exportChat';
import { formatRelativeTime } from '../utils/relativeTime';
import { pageLabel } from '../utils/pages';
import './UnansweredPanel.css';

interface CollectedPanelProps {
  quotes: Quote[];
  basket: BasketPassage[];
  /** The workspace the app is in: passages of another can't be asked about here. */
  workspace: string;
  onOpenPassage: (filename: string, chunkId: string | null, workspace: string) => void;
  onRemoveQuote: (id: string) => void;
  onRemovePassages: (chunkIds: string[]) => void;
  onClearBasket: () => void;
  /** Ask ``question`` with exactly these passages (``chunk_ids``). */
  onAskAbout: (question: string, chunkIds: string[]) => void;
  /** Scope the conversation to these documents. */
  onScopeTo: (filenames: string[]) => void;
  /** Changed to put focus in the basket's question box (passages were just added to
   *  ask about). */
  focusQuestionKey?: number;
}

function stamp(): string {
  return new Date().toISOString().slice(0, 10);
}

/** Quotes saved from answers and passages, and the basket of passages picked to ask
 *  about together. Both live in this browser. */
function CollectedPanel({
  quotes,
  basket,
  workspace,
  onOpenPassage,
  onRemoveQuote,
  onRemovePassages,
  onClearBasket,
  onAskAbout,
  onScopeTo,
  focusQuestionKey = 0,
}: CollectedPanelProps) {
  const questionRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (focusQuestionKey > 0) questionRef.current?.focus();
  }, [focusQuestionKey]);
  const [confirmClear, setConfirmClear] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);
  const [question, setQuestion] = useState('');
  const here = window.location;
  const usable = basket.filter((passage) => passage.workspace === workspace);
  const elsewhere = basket.length - usable.length;
  return (
    <>
      <section aria-labelledby="basket-heading" data-testid="basket">
        <h3 id="basket-heading" className="collected-panel__heading">
          Passages ({basket.length})
        </h3>
        {basket.length === 0 ? (
          <p className="unanswered-panel__message">
            Pick passages in Search, the viewer or an answer’s sources to ask about them together.
          </p>
        ) : (
          <>
            <form
              className="unanswered-panel__controls"
              onSubmit={(event) => {
                event.preventDefault();
                if (!question.trim() || usable.length === 0) return;
                onAskAbout(question.trim(), usable.map((passage) => passage.chunkId));
                setQuestion('');
              }}
            >
              <input
                ref={questionRef}
                className="unanswered-panel__filter"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Ask about these passages…"
                aria-label="Question about the passages in the basket"
                data-testid="basket-question"
              />
              <button
                type="submit"
                disabled={usable.length === 0 || !question.trim()}
                data-testid="basket-ask"
              >
                Ask
              </button>
            </form>
            <div className="unanswered-panel__controls">
              <button
                type="button"
                onClick={() => onScopeTo([...new Set(usable.map((passage) => passage.filename))])}
                disabled={usable.length === 0}
                data-testid="basket-scope"
              >
                Scope to their documents
              </button>
              <button
                type="button"
                onClick={() =>
                  downloadFile(`passages-${stamp()}.md`, 'text/markdown', basketToMarkdown(basket, here))
                }
                data-testid="basket-export"
              >
                Export
              </button>
              <button type="button" onClick={() => setConfirmClear(true)} data-testid="basket-clear">
                Clear
              </button>
            </div>
            {confirmClear && (
              <ConfirmInline
                prompt="Empty the basket?"
                confirmLabel="Empty"
                tone="main"
                onConfirm={() => {
                  setConfirmClear(false);
                  onClearBasket();
                }}
                onCancel={() => setConfirmClear(false)}
              />
            )}
            {elsewhere > 0 && (
              <p className="unanswered-panel__message" data-testid="basket-elsewhere">
                {elsewhere} {elsewhere === 1 ? 'passage is' : 'passages are'} from another
                workspace and left out of a question here.
              </p>
            )}
            <ul className="unanswered-panel__list">
              {basket.map((passage) => (
                <li key={passage.chunkId} className="unanswered-panel__row" data-testid="basket-row">
                  <span className="unanswered-panel__question" dir="auto">
                    {passage.filename}, {pageLabel(passage.page, passage.pageEnd)}
                    {passage.section ? ` · ${passage.section}` : ''}
                  </span>
                  <span className="unanswered-panel__meta" dir="auto">
                    {passage.snippet}
                  </span>
                  <span className="unanswered-panel__actions">
                    <button
                      type="button"
                      onClick={() => onOpenPassage(passage.filename, passage.chunkId, passage.workspace)}
                      aria-label={`Open the passage in ${passage.filename}`}
                    >
                      Open
                    </button>
                    <button
                      type="button"
                      onClick={() => onRemovePassages([passage.chunkId])}
                      aria-label={`Take the passage in ${passage.filename} out of the basket`}
                      data-testid="basket-remove"
                    >
                      Remove
                    </button>
                  </span>
                </li>
              ))}
            </ul>
          </>
        )}
      </section>
      <section aria-labelledby="quotes-heading" data-testid="quotes">
        <h3 id="quotes-heading" className="collected-panel__heading">
          Quotes ({quotes.length})
        </h3>
        {quotes.length === 0 ? (
          <p className="unanswered-panel__message">
            Select text in an answer or a passage and choose “Save quote”.
          </p>
        ) : (
          <>
            <div className="unanswered-panel__controls">
              <button
                type="button"
                onClick={() =>
                  downloadFile(`quotes-${stamp()}.md`, 'text/markdown', quotesToMarkdown(quotes, here))
                }
                data-testid="quotes-export-md"
              >
                Export Markdown
              </button>
              <button
                type="button"
                onClick={() => downloadFile(`quotes-${stamp()}.csv`, 'text/csv', quotesToCsv(quotes, here))}
                data-testid="quotes-export-csv"
              >
                Export CSV
              </button>
            </div>
            <ul className="unanswered-panel__list">
              {quotes.map((quote) => (
                <li key={quote.id} className="unanswered-panel__row" data-testid="quote-row">
                  <span className="unanswered-panel__question" dir="auto">
                    “{quote.text}”
                  </span>
                  <span className="unanswered-panel__meta mono">
                    <span>{quote.source ?? 'From an answer'}</span>
                    <span className="unanswered-panel__spacer" />
                    <span>{formatRelativeTime(quote.savedAt)}</span>
                  </span>
                  <span className="unanswered-panel__actions">
                    {quote.filename && (
                      <button
                        type="button"
                        onClick={() => onOpenPassage(quote.filename ?? '', quote.chunkId, quote.workspace)}
                        aria-label={`Open where the quote comes from (${quote.source ?? quote.filename})`}
                      >
                        Open
                      </button>
                    )}
                    <button
                      type="button"
                      onClick={() =>
                        void copyText(`> ${quote.text}${quote.source ? `\n— ${quote.source}` : ''}`).then(
                          (done) => setCopied(done ? quote.id : null),
                        )
                      }
                      data-testid="quote-copy"
                    >
                      {copied === quote.id ? 'Copied' : 'Copy'}
                    </button>
                    <button
                      type="button"
                      onClick={() => onRemoveQuote(quote.id)}
                      aria-label="Delete this quote"
                      data-testid="quote-remove"
                    >
                      Delete
                    </button>
                  </span>
                </li>
              ))}
            </ul>
          </>
        )}
      </section>
    </>
  );
}

export default memo(CollectedPanel);
