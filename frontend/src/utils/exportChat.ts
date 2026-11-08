import { usageTotal } from './deepLink';
import { formatCost } from './cost';
import type { CitationResponse } from '../api/types';
import type { ChatTurn } from '../components/ChatMessage';
import { toCsv } from './csv';
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
  const usage = usageTotal(turns);
  // What the answers used, at the end: an export is also how costs are reported.
  const footer = usage
    ? `\n\n---\n\n_Answers used ${usage.tokens.toLocaleString('en-US')} tokens${
        usage.costUsd !== undefined ? ` (about ${formatCost(usage.costUsd)})` : ''
      }._`
    : '';
  return (title ? `# ${title}\n\n${body}` : body) + footer;
}

/** Saved answers as one Markdown file: each with its question, conversation, sources
 *  and the quotes they rest on. */
export function savedAnswersToMarkdown(
  items: { conversationTitle: string; question: string | null; turn: ChatTurn }[],
): string {
  const blocks = items.map(({ conversationTitle, question, turn }) =>
    [
      `## ${question ?? 'Answer'}`,
      `_${conversationTitle} · ${DATE_TIME_FORMATTER.format(turn.timestamp)}_`,
      turn.content,
      citationLinesWithQuotes(turn.sources),
    ]
      .filter(Boolean)
      .join('\n\n'),
  );
  return `# Saved answers\n\n${blocks.join('\n\n---\n\n')}`;
}

export function chatToJson(turns: ChatTurn[]): string {
  return JSON.stringify(turns, null, 2);
}

/** A download's file name from free text (a conversation title): anything but word
 *  characters, dots and dashes becomes a dash. */
export function safeFilename(name: string): string {
  return name.replace(/[^\w.-]+/g, '-');
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

/** Every source cited in a conversation, one row each, for a spreadsheet: which answer
 *  (and question) cited it, and where it is — filename, page, section, chunk id — with
 *  the sentence it quotes, when citation spans found one. */
export function citedSourcesCsv(turns: ChatTurn[], title: string | undefined): string {
  const rows: unknown[][] = [];
  let answerNumber = 0;
  let question = '';
  for (const turn of turns) {
    if (turn.role === 'user') {
      question = turn.content;
      continue;
    }
    if (turn.role !== 'assistant' || !turn.content) continue;
    answerNumber += 1;
    for (const source of turn.sources) {
      rows.push([
        title ?? '',
        question,
        answerNumber,
        source.source_number,
        source.filename,
        pageLabel(source.page, source.page_end),
        source.section,
        source.chunk_id,
        source.spans?.map((span) => span.quote).join(' … ') ?? '',
      ]);
    }
  }
  return toCsv(
    ['conversation', 'question', 'answer', 'source', 'filename', 'page', 'section', 'chunk_id', 'quote'],
    rows,
  );
}
