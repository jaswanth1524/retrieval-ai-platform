import { createElement } from 'react';
import type { ComponentType } from 'react';
import { flushSync } from 'react-dom';
import { createRoot } from 'react-dom/client';
import type { Components } from 'react-markdown';
import type { CitationResponse } from '../api/types';
import type { ChatTurn } from '../components/ChatMessage';
import type { MarkdownProps } from '../components/PlainAnswer';
import { loadMarkdown } from '../components/markdownLoader';
import { DATE_TIME_FORMATTER, speakerLabel } from './exportChat';
import { pageLabel } from './pages';

export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// As in the app: an answer never loads an image (its alt text stands in), and links
// open elsewhere without handing the page over.
const EXPORT_COMPONENTS: Components = {
  img: ({ alt }) => (alt ? createElement('span', null, `[${alt}]`) : null),
  a: ({ href, children }) =>
    href && /^https?:/i.test(href)
      ? createElement('a', { href, rel: 'noopener noreferrer', target: '_blank' }, children)
      : createElement('span', null, children),
};

/** Markdown rendered to static HTML with the app's own (sanitizing) renderer: raw HTML
 *  in an answer stays text, as react-markdown never passes it through. */
function renderMarkdown(renderer: ComponentType<MarkdownProps>, text: string): string {
  const container = document.createElement('div');
  const root = createRoot(container);
  flushSync(() => root.render(createElement(renderer, { text, components: EXPORT_COMPONENTS })));
  const html = container.innerHTML;
  root.unmount();
  return html;
}

function sourcesHtml(sources: CitationResponse[]): string {
  if (sources.length === 0) return '';
  const items = sources
    .map((source) => {
      const where = `${escapeHtml(source.filename)} · ${escapeHtml(pageLabel(source.page, source.page_end))} · ${escapeHtml(source.section)}`;
      const quotes = (source.spans ?? [])
        .map((span) => `<blockquote dir="auto">${escapeHtml(span.quote)}</blockquote>`)
        .join('');
      return `<li><span class="n">[${source.source_number}]</span> ${where}${quotes}</li>`;
    })
    .join('');
  return `<ol class="sources">${items}</ol>`;
}

const STYLE = `
:root { color-scheme: light dark; --fg: #1d1d1f; --bg: #ffffff; --dim: #6e6e73; --rule: #d2d2d7; --accent: #2f6fde; }
@media (prefers-color-scheme: dark) { :root { --fg: #f5f5f7; --bg: #151517; --dim: #a1a1a6; --rule: #3a3a3c; --accent: #6ea8ff; } }
* { box-sizing: border-box; }
body { margin: 0 auto; max-width: 760px; padding: 32px 16px 64px; background: var(--bg); color: var(--fg); font: 15px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif; }
h1 { font-size: 22px; margin: 0 0 4px; }
.meta { color: var(--dim); font-size: 12px; margin: 0 0 24px; }
.turn { border-top: 1px solid var(--rule); padding: 16px 0; }
.who { font-size: 12px; font-weight: 600; color: var(--dim); text-transform: uppercase; letter-spacing: .04em; }
.who time { font-weight: 400; text-transform: none; margin-left: 6px; }
.turn.user .body { font-weight: 500; }
.turn.error .body { color: #c62828; }
.body table { border-collapse: collapse; display: block; overflow-x: auto; }
.body th, .body td { border: 1px solid var(--rule); padding: 4px 8px; text-align: left; }
.body pre { overflow-x: auto; padding: 8px; border: 1px solid var(--rule); border-radius: 6px; }
.sources { margin: 12px 0 0; padding-left: 0; list-style: none; font-size: 12px; color: var(--dim); }
.sources li { margin: 4px 0; }
.sources .n { color: var(--accent); font-weight: 600; }
blockquote { margin: 4px 0 4px 18px; padding-left: 8px; border-left: 2px solid var(--rule); font-style: italic; }
a { color: var(--accent); }
@media print { body { max-width: none; padding: 0; } .turn { break-inside: avoid; } a { color: inherit; } }
`;

/** A conversation as one self-contained HTML page (inline styles, no scripts, no
 *  requests): readable offline, printable, and safe to pass on. */
export async function chatToHtml(turns: ChatTurn[], title: string, exportedAt = Date.now()): Promise<string> {
  const renderer = await loadMarkdown();
  const body = turns
    .map((turn) => {
      const content =
        turn.role === 'assistant'
          ? renderMarkdown(renderer, turn.content)
          : `<p>${escapeHtml(turn.content).replace(/\n/g, '<br>')}</p>`;
      return [
        `<section class="turn ${turn.role}">`,
        `<div class="who">${speakerLabel(turn.role)}<time>${escapeHtml(DATE_TIME_FORMATTER.format(turn.timestamp))}</time></div>`,
        `<div class="body" dir="auto">${content}</div>`,
        sourcesHtml(turn.sources),
        '</section>',
      ].join('');
    })
    .join('\n');
  return [
    '<!doctype html>',
    '<html lang="en">',
    '<head>',
    '<meta charset="utf-8">',
    '<meta name="viewport" content="width=device-width, initial-scale=1">',
    // No scripts and nothing fetched: the page is only what is written here.
    `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">`,
    `<title>${escapeHtml(title)}</title>`,
    `<style>${STYLE}</style>`,
    '</head>',
    '<body>',
    `<h1 dir="auto">${escapeHtml(title)}</h1>`,
    `<p class="meta">Exported from DocRAG on ${escapeHtml(DATE_TIME_FORMATTER.format(exportedAt))}</p>`,
    body,
    '</body>',
    '</html>',
  ].join('\n');
}
