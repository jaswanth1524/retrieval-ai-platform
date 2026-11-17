import { WEAK_SUPPORT_MARGIN } from '../../appHelpers';
import { pageLabel } from '../../utils/pages';
import type { ChatTurn } from '../ChatMessage';

interface WhyThisAnswerProps {
  turn: ChatTurn;
  /** The server's weak-support line (its bar + WEAK_SUPPORT_MARGIN), when known. */
  weakSupportBelow?: number;
}

/** How strongly the best passage matched, in words. */
function strength(score: number, bar: number | undefined): string {
  if (bar === undefined) return '';
  if (score < bar) return ' — below the bar for being used';
  if (score < bar + WEAK_SUPPORT_MARGIN) return ' — only just over the bar';
  return ' — a clear match';
}

/** "Why this answer?" for readers (Engineer mode has the Retrieval and Trace tabs): what
 *  was searched, how well the best passage matched, and which passages it cites, in
 *  plain words — from what the answer already carries, with no request. */
export default function WhyThisAnswer({ turn, weakSupportBelow }: WhyThisAnswerProps) {
  const bar =
    typeof turn.rerankMinScore === 'number'
      ? turn.rerankMinScore
      : weakSupportBelow !== undefined
        ? weakSupportBelow - WEAK_SUPPORT_MARGIN
        : undefined;
  const documents = new Set(turn.sources.map((source) => source.filename));
  const read = turn.notFound ? (turn.contextSources ?? []) : [];
  return (
    <details className="chat-message__why" data-testid="chat-message-why">
      <summary>Why this answer?</summary>
      <ul>
        {turn.searchedAs && (
          <li>
            Read with the conversation, it was searched as “<span dir="auto">{turn.searchedAs}</span>”.
          </li>
        )}
        <li>{turn.scoped ? 'It looked only in the documents selected for this conversation.' : 'It looked in all your documents.'}</li>
        {turn.cached && <li>It was answered before: this is the stored answer (Regenerate asks again).</li>}
        {typeof turn.topRerankScore === 'number' && (
          <li data-testid="chat-message-why-score">
            The best passage found scored {turn.topRerankScore.toFixed(2)} out of 1
            {bar !== undefined ? ` (passages under ${bar.toFixed(2)} aren’t used)` : ''}
            {strength(turn.topRerankScore, bar)}.
          </li>
        )}
        {turn.weakMatch && (
          <li>No passage cleared that bar, so the answer was written from the closest ones, and the model was told they match only weakly.</li>
        )}
        {turn.notFound ? (
          <li>
            The model read {read.length > 0 ? `${read.length} passage${read.length === 1 ? '' : 's'}` : 'what was found'} and
            found no answer in {read.length === 1 ? 'it' : 'them'}.
          </li>
        ) : turn.sources.length > 0 ? (
          <li>
            It cites {turn.sources.length} passage{turn.sources.length === 1 ? '' : 's'} from {documents.size} document
            {documents.size === 1 ? '' : 's'}:
            <ul>
              {turn.sources.map((source) => (
                <li key={source.source_number} dir="auto">
                  [{source.source_number}] {source.filename}, {pageLabel(source.page, source.page_end)}
                  {source.section ? ` · ${source.section}` : ''}
                  {typeof source.rerank_score === 'number' ? ` · scored ${source.rerank_score.toFixed(2)}` : ''}
                </li>
              ))}
            </ul>
          </li>
        ) : (
          <li>It cites no passage.</li>
        )}
        {turn.autoCited && <li>The model cited nothing itself; its citations were added by matching its sentences to the passages.</li>}
        {(turn.passagesLeftOut ?? 0) > 0 && (
          <li>{turn.passagesLeftOut} of the passages picked for it didn’t fit and were left out.</li>
        )}
        {turn.model && <li>Written by {turn.model}.</li>}
      </ul>
    </details>
  );
}
