import type { CitationResponse } from '../../api/types';
import type { SourceChange } from '../../utils/citationFreshness';
import CitationCard from '../CitationCard';
import type { ChatTurn } from '../ChatMessage';
import { quotesOf } from './quotes';

interface AnswerSourcesProps {
  turn: ChatTurn;
  busy: boolean;
  onOpenSource?: (filename: string, chunkId: string, quotes?: string[]) => void;
  onCitationHover?: (citation: CitationResponse) => void;
  onCitationLeave?: () => void;
  sourceChangeOf?: (filename: string, answeredAtMs: number) => SourceChange | null;
  /** Rate a source not relevant (feedback on, answer finished). */
  onNotRelevant?: (source: CitationResponse) => void;
  /** Ask the question again without this source. */
  onAskWithout?: (source: CitationResponse) => void;
}

/** An answer's sources as cards, and for a not-found answer the passages it read. */
export default function AnswerSources({
  turn,
  busy,
  onOpenSource,
  onCitationHover,
  onCitationLeave,
  sourceChangeOf,
  onNotRelevant,
  onAskWithout,
}: AnswerSourcesProps) {
  return (
    <>
      {turn.contextSources && turn.contextSources.length > 0 && (
        <details className="chat-message__closest" data-testid="chat-message-closest">
          <summary>
            Passages it read ({turn.contextSources.length}) — none answered the question
          </summary>
          <div className="chat-message__citations">
            {turn.contextSources.map((source) => (
              <CitationCard
                key={`${turn.id}-context-${source.source_number}`}
                sourceNumber={source.source_number}
                filename={source.filename}
                page={source.page}
                pageEnd={source.page_end}
                section={source.section}
                onOpen={onOpenSource ? () => onOpenSource(source.filename, source.chunk_id) : undefined}
              />
            ))}
          </div>
        </details>
      )}
      {turn.sources.length > 0 && (
        <div className="chat-message__citations" data-testid="chat-message-sources">
          {turn.sources.map((source) => (
            <CitationCard
              key={`${turn.id}-${source.source_number}`}
              sourceNumber={source.source_number}
              filename={source.filename}
              page={source.page}
              pageEnd={source.page_end}
              section={source.section}
              onOpen={
                onOpenSource
                  ? () => onOpenSource(source.filename, source.chunk_id, quotesOf(source))
                  : undefined
              }
              onHoverStart={onCitationHover ? () => onCitationHover(source) : undefined}
              onHoverEnd={onCitationLeave}
              onNotRelevant={onNotRelevant ? () => onNotRelevant(source) : undefined}
              markedNotRelevant={turn.irrelevantSources?.includes(source.source_number)}
              score={source.rerank_score}
              change={sourceChangeOf?.(source.filename, turn.timestamp)}
              onAskWithout={onAskWithout ? () => onAskWithout(source) : undefined}
              askDisabled={busy}
            />
          ))}
        </div>
      )}
    </>
  );
}
