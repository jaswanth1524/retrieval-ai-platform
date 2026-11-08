import { useState } from 'react';
import type { Components } from 'react-markdown';
import type { AnswerVersion } from '../ChatMessage';
import CitationCard from '../CitationCard';
import { useMarkdownRenderer } from '../markdownLoader';
import { formatTurnTime } from '../../utils/exportChat';

interface AnswerVersionsProps {
  versions: AnswerVersion[];
  /** How the answer's markdown is rendered (the answer's own, without citation links). */
  components: Components;
  onOpenSource?: (filename: string, chunkId: string) => void;
}

/** The earlier answers to this question (Regenerate and Reformat keep them), shown one
 *  at a time under the current one: ‹ 1/2 › to step through them. */
export default function AnswerVersions({ versions, components, onOpenSource }: AnswerVersionsProps) {
  const [shown, setShown] = useState<number | null>(null);
  const Markdown = useMarkdownRenderer();
  if (versions.length === 0) return null;
  if (shown === null) {
    return (
      <button
        type="button"
        className="chat-message__action chat-message__versions-toggle"
        onClick={() => setShown(versions.length - 1)}
        data-testid="answer-versions-open"
      >
        Earlier answer{versions.length === 1 ? '' : 's'} ({versions.length})
      </button>
    );
  }
  const version = versions[shown];
  return (
    <section
      className="chat-message__versions"
      aria-label={`Earlier answer ${shown + 1} of ${versions.length}`}
      data-testid="answer-versions"
    >
      <div className="chat-message__versions-head">
        <button
          type="button"
          onClick={() => setShown(shown - 1)}
          disabled={shown === 0}
          aria-label="Older answer"
        >
          ‹
        </button>
        <span className="mono" data-testid="answer-versions-position">
          {shown + 1}/{versions.length}
        </span>
        <button
          type="button"
          onClick={() => setShown(shown + 1)}
          disabled={shown === versions.length - 1}
          aria-label="Newer answer"
        >
          ›
        </button>
        <span className="chat-message__versions-meta">
          {formatTurnTime(version.timestamp)}
          {version.model ? ` · ${version.model}` : ''}
        </span>
        <button type="button" onClick={() => setShown(null)} aria-label="Hide earlier answers">
          ✕
        </button>
      </div>
      <div className="chat-message__markdown" dir="auto" data-testid="answer-version-content">
        {Markdown ? <Markdown text={version.content} components={components} /> : <p className="chat-message__plain">{version.content}</p>}
      </div>
      {version.sources.length > 0 && (
        <div className="chat-message__citations">
          {version.sources.map((source) => (
            <CitationCard
              key={`v${shown}-${source.source_number}`}
              sourceNumber={source.source_number}
              filename={source.filename}
              page={source.page}
              pageEnd={source.page_end}
              section={source.section}
              onOpen={onOpenSource ? () => onOpenSource(source.filename, source.chunk_id) : undefined}
            />
          ))}
        </div>
      )}
    </section>
  );
}
