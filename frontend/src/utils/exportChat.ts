import type { ChatTurn } from '../components/ChatMessage';

const TIME_FORMATTER = new Intl.DateTimeFormat(undefined, {
  hour: '2-digit',
  minute: '2-digit',
});

function speakerLabel(role: ChatTurn['role']): string {
  if (role === 'user') return 'You';
  if (role === 'assistant') return 'DocRAG';
  return 'Error';
}

export function chatToMarkdown(turns: ChatTurn[]): string {
  const blocks = turns.map((turn) => {
    const heading = `## ${speakerLabel(turn.role)} (${TIME_FORMATTER.format(turn.timestamp)})`;
    const sources = turn.sources
      .map((s) => `[${s.source_number}] ${s.filename} · p.${s.page} · ${s.section}`)
      .join('\n');
    return [heading, turn.content, sources].filter(Boolean).join('\n\n');
  });
  return blocks.join('\n\n---\n\n');
}

export function chatToJson(turns: ChatTurn[]): string {
  return JSON.stringify(turns, null, 2);
}

export function downloadFile(filename: string, mimeType: string, content: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}
