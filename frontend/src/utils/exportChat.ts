import type { CitationResponse } from '../api/types';
import type { ChatTurn } from '../components/ChatMessage';
import { pageLabel } from './pages';

export const TIME_FORMATTER = new Intl.DateTimeFormat(undefined, {
  hour: '2-digit',
  minute: '2-digit',
});

export const DATE_TIME_FORMATTER = new Intl.DateTimeFormat(undefined, {
  year: 'numeric',
  month: 'short',
  day: 'numeric',
  hour: '2-digit',
  minute: '2-digit',
});

/** A message's time, with its date unless it is from today: a conversation reopened
 * days later showed "09:30" on every turn with nothing saying which day. */
export function formatTurnTime(timestamp: number, now = Date.now()): string {
  return new Date(timestamp).toDateString() === new Date(now).toDateString()
    ? TIME_FORMATTER.format(timestamp)
    : DATE_TIME_FORMATTER.format(timestamp);
}

export function speakerLabel(role: ChatTurn['role']): string {
  if (role === 'user') return 'You';
  if (role === 'assistant') return 'DocRAG';
  return 'Error';
}

function citationLines(sources: CitationResponse[]): string {
  return sources
    .map((s) => `[${s.source_number}] ${s.filename} · ${pageLabel(s.page, s.page_end)} · ${s.section}`)
    .join('\n');
}

/** `content` followed by its sources as footnotes — what Copy puts on the clipboard,
 * so the answer's [n] markers don't dangle. Same lines the Markdown export writes. */
export function withCitationFootnotes(content: string, sources: CitationResponse[]): string {
  return [content, citationLines(sources)].filter(Boolean).join('\n\n');
}

/** Each source line followed by the passages its answer rests on, as quotes. */
function citationLinesWithQuotes(sources: CitationResponse[]): string {
  return sources
    .map((source) => {
      const line = `[${source.source_number}] ${source.filename} · ${pageLabel(source.page, source.page_end)} · ${source.section}`;
      const quotes = (source.spans ?? []).map((span) => `> ${span.quote.replace(/\s+/g, ' ').trim()}`);
      return [line, ...quotes].join('\n');
    })
    .join('\n');
}

/** A conversation as Markdown: its title, then each turn (dated), its sources and the
 *  quotes they rest on. */
export function chatToMarkdown(turns: ChatTurn[], title?: string): string {
  const blocks = turns.map((turn) => {
    // Always dated: an export is read later, away from the conversation.
    const heading = `## ${speakerLabel(turn.role)} (${DATE_TIME_FORMATTER.format(turn.timestamp)})`;
    return [heading, turn.content, citationLinesWithQuotes(turn.sources)]
      .filter(Boolean)
      .join('\n\n');
  });
  const body = blocks.join('\n\n---\n\n');
  return title ? `# ${title}\n\n${body}` : body;
}

export function chatToJson(turns: ChatTurn[]): string {
  return JSON.stringify(turns, null, 2);
}

export function downloadFile(filename: string, mimeType: string, content: string | Blob): void {
  const blob = content instanceof Blob ? content : new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  // Some browsers ignore a click on a detached anchor, and revoking the URL in the
  // same tick can cancel the download before it starts.
  anchor.style.display = 'none';
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  // Not revoked at once: Safari can still be reading a large file (a corpus backup)
  // when the click returns. FileSaver.js waits 40 s for the same reason.
  setTimeout(() => URL.revokeObjectURL(url), 40_000);
}
