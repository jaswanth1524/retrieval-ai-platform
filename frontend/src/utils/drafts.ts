import { MAX_QUESTION_CHARS } from './limits';
import { readStoredJson, writeStored } from './safeStorage';

// Unsent questions, per conversation, so a reload (or a switch away and back) doesn't
// lose what was being typed. Best-effort like every preference: localStorage may be
// unavailable, and then a draft simply lives as long as the page.
const DRAFTS_STORAGE_KEY = 'docrag-drafts';
const MAX_DRAFTS = 50;
/** The draft key while no conversation exists yet. */
export const NEW_CONVERSATION_DRAFT = 'new';

type Drafts = Record<string, { text: string; at: number }>;

function readDrafts(): Drafts {
  const stored = readStoredJson<unknown>(DRAFTS_STORAGE_KEY);
  if (!stored || typeof stored !== 'object' || Array.isArray(stored)) return {};
  const drafts: Drafts = {};
  for (const [key, value] of Object.entries(stored as Record<string, unknown>)) {
    const entry = value as { text?: unknown; at?: unknown } | null;
    if (entry && typeof entry.text === 'string' && typeof entry.at === 'number') {
      drafts[key] = { text: entry.text, at: entry.at };
    }
  }
  return drafts;
}

export function readDraft(key: string): string {
  return readDrafts()[key]?.text ?? '';
}

/** Save (or, for blank text, forget) the draft for ``key``; the oldest go past the cap. */
export function writeDraft(key: string, text: string): void {
  const drafts = readDrafts();
  if (!text.trim()) {
    if (!(key in drafts)) return;
    delete drafts[key];
  } else {
    drafts[key] = { text: text.slice(0, MAX_QUESTION_CHARS), at: Date.now() };
    const keys = Object.keys(drafts).sort((a, b) => drafts[b].at - drafts[a].at);
    for (const old of keys.slice(MAX_DRAFTS)) delete drafts[old];
  }
  writeStored(DRAFTS_STORAGE_KEY, Object.keys(drafts).length ? JSON.stringify(drafts) : null);
}


/** Forget the drafts of conversations that no longer exist (deleted here or in another
 *  tab). ``keep``: the ids still listed; the "new" key is always kept. */
export function pruneDrafts(keep: ReadonlySet<string>): void {
  const drafts = readDrafts();
  const stale = Object.keys(drafts).filter((key) => key !== NEW_CONVERSATION_DRAFT && !keep.has(key));
  if (stale.length === 0) return;
  for (const key of stale) delete drafts[key];
  writeStored(DRAFTS_STORAGE_KEY, Object.keys(drafts).length ? JSON.stringify(drafts) : null);
}

