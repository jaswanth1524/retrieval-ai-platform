import type { QueryLogEntryResponse } from '../api/types';

// The query log's status for "the documents didn't answer it" (api/query_log.py).
export const UNCOVERED = 'insufficient_context';

/** The same question asked several times: one row, with how often and when last. */
export interface AskedQuestion {
  question: string;
  count: number;
  /** The latest time it was asked, and that ask's details. */
  latest: QueryLogEntryResponse;
  /** How many of its asks the documents didn't answer. */
  unanswered: number;
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
