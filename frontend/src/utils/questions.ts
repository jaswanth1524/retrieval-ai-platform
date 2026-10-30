import type { QueryLogEntryResponse } from '../api/types';

// The query log's status for "the documents didn't answer it" (api/query_log.py).
export const UNCOVERED = 'insufficient_context';
// The most GET /query-log returns (QUERY_LOG_LIST_MAX).
export const QUERY_LOG_FETCH_LIMIT = 1000;

/** The same question asked several times: one row, with how often and when last. */
export interface AskedQuestion {
  question: string;
  count: number;
  /** The latest time it was asked, and that ask's details. */
  latest: QueryLogEntryResponse;
  /** How many of its asks the documents didn't answer. */
  unanswered: number;
}

/** Not answered the last time it was asked: a question missed once and answered since
 *  (a document was added) no longer shows what the corpus is missing. */
export function stillUnanswered(row: AskedQuestion): boolean {
  return row.latest.status === UNCOVERED;
}

export function normalized(question: string): string {
  return question.trim().replace(/\s+/g, ' ').toLowerCase();
}

/** ``entries`` (newest first) grouped by question, case and spacing ignored. */
export function groupQuestions(entries: readonly QueryLogEntryResponse[]): AskedQuestion[] {
  const groups = new Map<string, AskedQuestion>();
  for (const entry of entries) {
    const key = normalized(entry.question);
    const group = groups.get(key);
    const missed = entry.status === UNCOVERED ? 1 : 0;
    if (group) {
      group.count += 1;
      group.unanswered += missed;
    } else {
      groups.set(key, { question: entry.question, count: 1, latest: entry, unanswered: missed });
    }
  }
  return [...groups.values()];
}

function entryKey(entry: QueryLogEntryResponse): string {
  return `${entry.created_at}\u0000${entry.trace_id ?? ''}\u0000${entry.question}`;
}

/** The latest ``recent`` asks plus older unanswered ones (``uncovered``), newest first,
 *  each once. Every ask in the recent window that wasn't answered is in both lists, and
 *  every extra one from ``uncovered`` is older than the whole window — so the first
 *  ``recent.length`` entries are exactly ``recent``. */
export function mergeQueryLogs(
  recent: readonly QueryLogEntryResponse[],
  uncovered: readonly QueryLogEntryResponse[],
): QueryLogEntryResponse[] {
  const seen = new Set(recent.map(entryKey));
  const older = uncovered.filter((entry) => !seen.has(entryKey(entry)));
  return [...recent, ...older].sort((a, b) => b.created_at - a.created_at);
}
