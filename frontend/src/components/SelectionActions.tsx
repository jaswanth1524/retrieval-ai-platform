import { useEffect, useRef, useState } from 'react';
import { copyText } from '../utils/clipboard';
import { MAX_QUESTION_CHARS } from '../utils/limits';

/** Where a selection may be acted on: an answer's text, or a passage in the viewer
 *  (whose element names its source in ``data-quote-source``). */
const SELECTABLE = '.chat-message__markdown, [data-quote-source]';
// A selection shorter than this is a click that moved, not a quote.
const MIN_SELECTION_CHARS = 3;
const MAX_SELECTION_CHARS = 2000;

interface Selected {
  text: string;
  source: string | null;
  top: number;
  left: number;
}

/** A small bar over text selected in an answer or a viewer passage: ask about it (into
 *  the composer, not sent), search for passages like it, or copy it as a quote with
 *  where it comes from. Keyboard users get it too: it follows the selection, not the
 *  mouse. */
export default function SelectionActions({
  onAsk,
  onSearch,
}: {
  onAsk: (text: string) => void;
  onSearch: (text: string) => void;
}) {
  const [selected, setSelected] = useState<Selected | null>(null);
  const [copied, setCopied] = useState(false);
  const barRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let frame = 0;
    const read = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const selection = document.getSelection();
        const text = selection?.toString().replace(/\s+/g, ' ').trim() ?? '';
        if (!selection || selection.rangeCount === 0 || text.length < MIN_SELECTION_CHARS) {
          // A click on the bar itself clears the selection first: keep the bar then.
          if (!barRef.current?.contains(document.activeElement)) setSelected(null);
          return;
        }
        const range = selection.getRangeAt(0);
        const start = range.startContainer instanceof Element ? range.startContainer : range.startContainer.parentElement;
        const host = start?.closest(SELECTABLE);
        if (!host || !host.contains(range.endContainer)) {
          setSelected(null);
          return;
        }
        const box = range.getBoundingClientRect();
        setCopied(false);
        setSelected({
          text: text.slice(0, MAX_SELECTION_CHARS),
          source: host.getAttribute('data-quote-source'),
          top: Math.max(8, box.top - 44),
          left: Math.min(Math.max(8, box.left), window.innerWidth - 280),
        });
      });
    };
    document.addEventListener('selectionchange', read);
    return () => {
      cancelAnimationFrame(frame);
      document.removeEventListener('selectionchange', read);
    };
  }, []);

  if (!selected) return null;
  const quote = `> ${selected.text}${selected.source ? `\n— ${selected.source}` : ''}`;
  return (
    <div
      ref={barRef}
      className="selection-actions"
      role="toolbar"
      aria-label="Selected text"
      style={{ top: selected.top, left: selected.left }}
      data-testid="selection-actions"
    >
      <button
        type="button"
        onMouseDown={(event) => event.preventDefault()}
        onClick={() => {
          onAsk(`About “${selected.text.slice(0, MAX_QUESTION_CHARS - 40)}”: `);
          setSelected(null);
        }}
        data-testid="selection-ask"
      >
        Ask about this
      </button>
      <button
        type="button"
        onMouseDown={(event) => event.preventDefault()}
        onClick={() => {
          onSearch(selected.text.slice(0, MAX_QUESTION_CHARS));
          setSelected(null);
        }}
        data-testid="selection-search"
      >
        Similar passages
      </button>
      <button
        type="button"
        onMouseDown={(event) => event.preventDefault()}
        onClick={() => void copyText(quote).then(setCopied)}
        data-testid="selection-copy"
      >
        {copied ? 'Copied' : 'Copy as quote'}
      </button>
    </div>
  );
}
