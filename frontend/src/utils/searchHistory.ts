import { getWorkspace } from '../api/client';
import type { SearchResultResponse } from '../api/types';

/** The Search panel's recent searches, remembered in this browser (newest first), per
 *  workspace: a search of another workspace's documents isn't one to pick here. */
export const RECENT_SEARCHES_KEY = 'docrag-recent-searches';
export const MAX_RECENT_SEARCHES = 8;
const MAX_REMEMBERED_CHARS = 200;

/** Where the current workspace's recent searches are kept (the default workspace keeps
 *  the key an earlier release used). */
export function recentSearchesKey(workspace: string = getWorkspace()): string {
  return workspace && workspace !== 'default' ? `${RECENT_SEARCHES_KEY}:${workspace}` : RECENT_SEARCHES_KEY;
}

export function clearRecentSearches(): string[] {
  try {
    localStorage.removeItem(recentSearchesKey());
  } catch {
    // Nothing was kept.
  }
  return [];
}

export function readRecentSearches(): string[] {
  try {
    const raw: unknown = JSON.parse(localStorage.getItem(recentSearchesKey()) ?? '[]');
    return Array.isArray(raw)
      ? raw.filter((item): item is string => typeof item === 'string' && item.length <= 500).slice(0, MAX_RECENT_SEARCHES)
      : [];
  } catch {
    return [];
  }
}

/** ``query`` first in the recent searches (once); the list as it now is. */
export function rememberSearch(query: string): string[] {
  const text = query.trim();
  // A whole selected passage sent to "Similar passages" isn't something to pick again.
  if (text.length > MAX_REMEMBERED_CHARS) return readRecentSearches();
  const next = [text, ...readRecentSearches().filter((item) => item !== text)]
    .filter(Boolean)
    .slice(0, MAX_RECENT_SEARCHES);
  try {
    localStorage.setItem(recentSearchesKey(), JSON.stringify(next));
  } catch {
    // Private mode: this session only.
  }
  return next;
}

/** ``results`` with each document's passages together, documents in the order of their
 *  best passage (the ranking otherwise kept). */
export function groupByDocument(results: SearchResultResponse[]): SearchResultResponse[] {
  const order: string[] = [];
  const byFile = new Map<string, SearchResultResponse[]>();
  for (const result of results) {
    if (!byFile.has(result.filename)) {
      order.push(result.filename);
      byFile.set(result.filename, []);
    }
    byFile.get(result.filename)?.push(result);
  }
  return order.flatMap((name) => byFile.get(name) ?? []);
}
