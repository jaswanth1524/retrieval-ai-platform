import { useEffect, useRef, useState } from 'react';
import { copyText } from '../utils/clipboard';
import { MAX_QUESTION_CHARS } from '../utils/limits';

/** Where a selection may be acted on: an answer's text, or a passage in the viewer
 *  (whose element names its source in ``data-quote-source``). */
const SELECTABLE = '.chat-message__markdown, [data-quote-source]';
// A selection shorter than this is a click that moved, not a quote.
const MIN_SELECTION_CHARS = 3;
const MAX_SELECTION_CHARS = 2000;

// The bar's size, for keeping it on screen (three buttons, or wrapped on a phone).
const BAR_WIDTH = 400;
const BAR_HEIGHT = 36;

/** The range's text without the citation buttons' digits ("retrieval 1 …"), spacing
 *  collapsed. */
function selectedText(range: Range): string {
  const copy = range.cloneContents();
  copy.querySelectorAll('[data-testid="inline-citation"]').forEach((marker) => marker.remove());
  return (copy.textContent ?? '').replace(/\s+/g, ' ').trim();
}

/** A quote to keep: its text and, from a viewer passage, where it comes from. */
export interface SelectedQuote {
  text: string;
  source: string | null;
  filename: string | null;
  chunkId: string | null;
}

interface Selected {
  text: string;
  source: string | null;
  filename: string | null;
  chunkId: string | null;
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
  onSave,
}: {
  onAsk: (text: string) => void;
  onSearch: (text: string) => void;
  /** Keep the selection as a quote (the Quotes & passages panel). */
  onSave?: (quote: SelectedQuote) => void;
}) {
  const [saved, setSaved] = useState(false);
  const [selected, setSelected] = useState<Selected | null>(null);
  const [copied, setCopied] = useState(false);
  const barRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let frame = 0;
    const read = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const selection = document.getSelection();
        // Text selected inside a field (the viewer's "ask about this passage" box) is
        // the reader's own typing, not a quote.
        const active = document.activeElement;
        if (active instanceof HTMLInputElement || active instanceof HTMLTextAreaElement) {
          setSelected(null);
          return;
        }
        const text = selection && selection.rangeCount > 0 ? selectedText(selection.getRangeAt(0)) : '';
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
        // A window narrower than the bar wraps it to two rows.
        const barHeight = window.innerWidth < BAR_WIDTH + 16 ? BAR_HEIGHT * 2 : BAR_HEIGHT;
        setCopied(false);
        setSaved(false);
        setSelected({
          text: text.slice(0, MAX_SELECTION_CHARS),
          source: host.getAttribute('data-quote-source'),
          filename: host.getAttribute('data-quote-filename'),
          chunkId: host.getAttribute('data-chunk-id'),
          // Above the selection; below it when there is no room above (the bar would
          // cover the text it is about).
          top: box.top >= barHeight + 12 ? box.top - barHeight - 8 : box.bottom + 8,
          left: Math.max(8, Math.min(box.left, window.innerWidth - BAR_WIDTH - 8)),
        });
      });
    };
    // Escape puts the bar away (the selection stays, to copy the usual way).
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setSelected(null);
    };
    document.addEventListener('selectionchange', read);
    // The bar is fixed to the window: it follows the text when the thread or the
    // viewer scrolls, and when the window is resized.
    document.addEventListener('scroll', read, true);
    window.addEventListener('resize', read);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      cancelAnimationFrame(frame);
      document.removeEventListener('selectionchange', read);
      document.removeEventListener('scroll', read, true);
      window.removeEventListener('resize', read);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, []);
  // "Copied" is said for a moment, not until the next selection.
  useEffect(() => {
    if (!copied) return;
    const timer = setTimeout(() => setCopied(false), 2000);
    return () => clearTimeout(timer);
  }, [copied]);

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
      {onSave && (
        <button
          type="button"
          onMouseDown={(event) => event.preventDefault()}
          onClick={() => {
            onSave({
              text: selected.text,
              source: selected.source,
              filename: selected.filename,
              chunkId: selected.chunkId,
            });
            setSaved(true);
          }}
          disabled={saved}
          data-testid="selection-save"
        >
          {saved ? 'Saved' : 'Save quote'}
        </button>
      )}
      <span className="visually-hidden" role="status">
        {copied ? 'Quote copied.' : saved ? 'Quote saved.' : ''}
      </span>
    </div>
  );
}
