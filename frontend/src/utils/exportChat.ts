import type { CitationResponse } from '../api/types';
import type { ChatTurn } from '../components/ChatMessage';

export const TIME_FORMATTER = new Intl.DateTimeFormat(undefined, {
  hour: '2-digit',
  minute: '2-digit',
});

function speakerLabel(role: ChatTurn['role']): string {
  if (role === 'user') return 'You';
  if (role === 'assistant') return 'DocRAG';
  return 'Error';
}

function citationLines(sources: CitationResponse[]): string {
  return sources
    .map((s) => `[${s.source_number}] ${s.filename} · p.${s.page} · ${s.section}`)
    .join('\n');
}

/** `content` followed by its sources as footnotes — what Copy puts on the clipboard,
 * so the answer's [n] markers don't dangle. Same lines the Markdown export writes. */
export function withCitationFootnotes(content: string, sources: CitationResponse[]): string {
  return [content, citationLines(sources)].filter(Boolean).join('\n\n');
}

export function chatToMarkdown(turns: ChatTurn[]): string {
  const blocks = turns.map((turn) => {
    const heading = `## ${speakerLabel(turn.role)} (${TIME_FORMATTER.format(turn.timestamp)})`;
    return [heading, withCitationFootnotes(turn.content, turn.sources)]
      .filter(Boolean)
      .join('\n\n');
  });
  return blocks.join('\n\n---\n\n');
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
