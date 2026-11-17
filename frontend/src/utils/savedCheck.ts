import type { SearchRequest, SearchResponse } from '../api/types';
import type { SavedAnswer } from '../hooks/useChat';

/** What searching a saved answer's question again found (no LLM: POST /search):
 *  - ``found``: a passage it cited still ranks among the results;
 *  - ``moved``: none of its passages ranks any more (re-indexed, changed or deleted
 *    documents, or other documents now outrank them) — worth asking again;
 *  - ``weak``: nothing clears the relevance bar any more. */
export type SavedCheck = 'found' | 'moved' | 'weak';

export const SAVED_CHECK_TOP_K = 8;

export const SAVED_CHECK_LABELS: Record<SavedCheck, string> = {
  found: 'Still found: its passages rank for the question.',
  moved: 'Its passages no longer rank for the question — ask it again.',
  weak: 'Only weak matches now — the documents may have changed.',
};

/** Whether ``item`` can be checked: a question (not a summary, whose "question" is a
 *  label; not a follow-up, which was searched as the server rewrote it), cited
 *  passages, this workspace's. */
export function checkable(item: SavedAnswer, workspace: string): boolean {
  return (
    Boolean(item.question) &&
    !item.task &&
    !item.turn.followUp &&
    item.turn.sources.length > 0 &&
    item.workspace === workspace
  );
}

/** The search that looks where the answer's question looked: its page/section/date
 *  narrowing, and — asked within some documents or tags (which aren't kept on the
 *  turn) — within the documents it cited. Picked passages aren't a search at all. */
export function savedCheckRequest(item: SavedAnswer): SearchRequest {
  const { chunk_ids: _picked, ...narrowing } = item.narrowing ?? {};
  const cited = [...new Set(item.turn.sources.map((source) => source.filename))];
  return {
    query: item.question ?? '',
    rerank_top_k: SAVED_CHECK_TOP_K,
    ...narrowing,
    ...(item.turn.scoped ? { filenames: cited.slice(0, 20) } : {}),
  };
}

export function judgeSaved(item: SavedAnswer, response: SearchResponse): SavedCheck {
  if (response.weak_matches || response.results.length === 0) return 'weak';
  const cited = new Set(item.turn.sources.map((source) => source.chunk_id));
  return response.results.some((result) => cited.has(result.chunk_id)) ? 'found' : 'moved';
}
