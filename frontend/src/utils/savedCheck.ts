import type { RecheckRequest, RecheckResponse, SearchRequest, SearchResponse } from '../api/types';
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

/** POST /questions/recheck's body for a saved answer: its text and cited passages. */
export function recheckRequest(item: SavedAnswer): RecheckRequest {
  return {
    answer: item.turn.content.slice(0, 8000),
    sources: item.turn.sources.slice(0, 50).map(({ source_number, filename, chunk_id }) => ({
      source_number,
      filename,
      chunk_id,
    })),
  };
}

/** One saved answer among them all. */
export function savedKey(item: SavedAnswer): string {
  return `${item.conversationId}:${item.turn.id}`;
}

/** What a recheck found wrong: a cited document gone, or a sentence it read that is
 *  no longer backed. Not ``still_holds``: that is false for any answer longer than one
 *  recheck reads (``more``). */
export function noLongerHolds(response: RecheckResponse): boolean {
  return (
    response.sources.some((source) => source.state === 'deleted') ||
    response.sentences.some(
      (sentence) => sentence.status === 'unsupported' || sentence.status === 'numbers_differ',
    )
  );
}

/** A recheck in one line: holds, or what changed and what no longer is backed. */
export function recheckSummary(response: RecheckResponse): string {
  if (response.still_holds) return 'Still holds: its passages back every sentence.';
  if (response.more && !noLongerHolds(response)) {
    return `Holds so far: its first ${response.sentences.length} sentences are backed (the rest weren’t checked).`;
  }
  const count = (state: string) => response.sources.filter((s) => s.state === state).length;
  const unbacked = response.sentences.filter(
    (s) => s.status === 'unsupported' || s.status === 'numbers_differ',
  ).length;
  const parts = [
    unbacked ? `${unbacked} sentence${unbacked === 1 ? '' : 's'} no longer backed` : '',
    count('changed') ? `${count('changed')} cited passage${count('changed') === 1 ? '' : 's'} changed` : '',
    count('deleted') ? `${count('deleted')} cited document${count('deleted') === 1 ? '' : 's'} deleted` : '',
    response.more ? `only its first ${response.sentences.length} sentences were checked` : '',
  ].filter(Boolean);
  return `Check it: ${parts.join(', ')}.`;
}
