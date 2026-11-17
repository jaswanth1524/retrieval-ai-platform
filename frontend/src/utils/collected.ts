/** What the reader collects while reading, kept in this browser: quotes (text selected
 *  in an answer or a passage, with where it comes from) and a basket of passages picked
 *  in Search, the viewer or an answer's sources — to ask about together, scope a
 *  question to their documents, or export. Each list is capped; the oldest go first. */
import { toCsv } from './csv';
import { passageLinkUrl } from './deepLink';

export const QUOTES_STORAGE_KEY = 'docrag-quotes';
export const BASKET_STORAGE_KEY = 'docrag-basket';
export const MAX_QUOTES = 500;
// The server takes this many picked passages in one question (REQUEST_CHUNK_IDS_MAX).
export const MAX_BASKET = 50;
const MAX_QUOTE_CHARS = 2000;
const MAX_FIELD_CHARS = 1024;

export interface Quote {
  id: string;
  text: string;
  /** "guide.pdf, p. 3, Setup" — as the selection bar shows it; null for an answer. */
  source: string | null;
  filename: string | null;
  chunkId: string | null;
  /** The workspace the passage is in ("" = the default one). */
  workspace: string;
  savedAt: number;
}

export interface BasketPassage {
  chunkId: string;
  filename: string;
  page: number;
  pageEnd?: number | null;
  section: string;
  /** The start of the passage's text, to recognise it in the list. */
  snippet: string;
  workspace: string;
  addedAt: number;
}

function text(value: unknown, max = MAX_FIELD_CHARS): string | null {
  return typeof value === 'string' && value.length > 0 ? value.slice(0, max) : null;
}

function readList(key: string): unknown[] {
  try {
    const raw: unknown = JSON.parse(localStorage.getItem(key) ?? '[]');
    return Array.isArray(raw) ? raw : [];
  } catch {
    return [];
  }
}

function writeList(key: string, items: unknown[]): boolean {
  try {
    localStorage.setItem(key, JSON.stringify(items));
    return true;
  } catch {
    return false;
  }
}

export function readQuotes(): Quote[] {
  const quotes: Quote[] = [];
  for (const raw of readList(QUOTES_STORAGE_KEY)) {
    if (!raw || typeof raw !== 'object') continue;
    const item = raw as Record<string, unknown>;
    const id = text(item.id, 64);
    const body = text(item.text, MAX_QUOTE_CHARS);
    if (!id || !body || typeof item.savedAt !== 'number') continue;
    quotes.push({
      id,
      text: body,
      source: text(item.source),
      filename: text(item.filename),
      chunkId: text(item.chunkId, 64),
      workspace: text(item.workspace, 64) ?? '',
      savedAt: item.savedAt,
    });
  }
  return quotes.sort((a, b) => b.savedAt - a.savedAt);
}

/** Saved newest first; the same text from the same place twice is kept once. */
export function saveQuote(quote: Quote): Quote[] {
  const others = readQuotes().filter(
    (kept) => !(kept.text === quote.text && kept.chunkId === quote.chunkId),
  );
  const next = [{ ...quote, text: quote.text.slice(0, MAX_QUOTE_CHARS) }, ...others].slice(
    0,
    MAX_QUOTES,
  );
  writeList(QUOTES_STORAGE_KEY, next);
  return next;
}

export function removeQuote(id: string): Quote[] {
  const next = readQuotes().filter((quote) => quote.id !== id);
  writeList(QUOTES_STORAGE_KEY, next);
  return next;
}

export function readBasket(): BasketPassage[] {
  const passages: BasketPassage[] = [];
  for (const raw of readList(BASKET_STORAGE_KEY)) {
    if (!raw || typeof raw !== 'object') continue;
    const item = raw as Record<string, unknown>;
    const chunkId = text(item.chunkId, 64);
    const filename = text(item.filename);
    if (!chunkId || !filename || typeof item.page !== 'number' || typeof item.addedAt !== 'number') {
      continue;
    }
    passages.push({
      chunkId,
      filename,
      page: item.page,
      pageEnd: typeof item.pageEnd === 'number' ? item.pageEnd : null,
      section: text(item.section, 300) ?? '',
      snippet: text(item.snippet, 300) ?? '',
      workspace: text(item.workspace, 64) ?? '',
      addedAt: item.addedAt,
    });
  }
  return passages;
}

/** Added at the end (kept in the order picked); one already there is left as it is.
 *  Past MAX_BASKET, the oldest go. */
export function addToBasket(added: BasketPassage[]): BasketPassage[] {
  const current = readBasket();
  const known = new Set(current.map((passage) => passage.chunkId));
  const next = [...current, ...added.filter((passage) => !known.has(passage.chunkId))].slice(
    -MAX_BASKET,
  );
  writeList(BASKET_STORAGE_KEY, next);
  return next;
}

export function removeFromBasket(chunkIds: string[]): BasketPassage[] {
  const gone = new Set(chunkIds);
  const next = readBasket().filter((passage) => !gone.has(passage.chunkId));
  writeList(BASKET_STORAGE_KEY, next);
  return next;
}

export function clearBasket(): BasketPassage[] {
  writeList(BASKET_STORAGE_KEY, []);
  return [];
}

type Here = Pick<Location, 'origin' | 'pathname'>;

function link(here: Here, filename: string | null, chunkId: string | null, workspace: string) {
  return filename && chunkId ? passageLinkUrl({ filename, chunkId, workspace }, here) : '';
}

/** The quotes as Markdown: each a block quote, its source and a link to the passage. */
export function quotesToMarkdown(quotes: Quote[], here: Here): string {
  return quotes
    .map((quote) => {
      const url = link(here, quote.filename, quote.chunkId, quote.workspace);
      const where = quote.source ?? 'an answer';
      return `> ${quote.text.replace(/\n/g, '\n> ')}\n\n— ${url ? `[${where}](${url})` : where}`;
    })
    .join('\n\n');
}

export function quotesToCsv(quotes: Quote[], here: Here): string {
  return toCsv(
    ['quote', 'source', 'link', 'saved'],
    quotes.map((quote) => [
      quote.text,
      quote.source ?? '',
      link(here, quote.filename, quote.chunkId, quote.workspace),
      new Date(quote.savedAt).toISOString(),
    ]),
  );
}

export function basketToMarkdown(passages: BasketPassage[], here: Here): string {
  return passages
    .map((passage) => {
      const url = link(here, passage.filename, passage.chunkId, passage.workspace);
      const page = passage.pageEnd ? `p. ${passage.page}-${passage.pageEnd}` : `p. ${passage.page}`;
      return `- [${passage.filename}, ${page}${passage.section ? `, ${passage.section}` : ''}](${url}) — ${passage.snippet}`;
    })
    .join('\n');
}
