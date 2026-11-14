import { useEffect, useMemo, useRef, useState } from 'react';
import type { Components } from 'react-markdown';
import type { AnswerVersion } from '../ChatMessage';
import CitationCard from '../CitationCard';
import { useMarkdownRenderer } from '../markdownLoader';
import { styleLabel } from '../../utils/answerVersions';
import { formatTurnTime } from '../../utils/exportChat';
import { wordDiff } from '../../utils/textDiff';

interface AnswerVersionsProps {
  versions: AnswerVersion[];
  /** How the answer's markdown is rendered (the answer's own, without citation links). */
  components: Components;
  onOpenSource?: (filename: string, chunkId: string) => void;
  /** The answer shown now: what a version is compared with. */
  current?: string;
  /** Make version ``index`` the answer again (the current one becomes a version). */
  onUse?: (index: number) => void;
}

/** The earlier answers to this question (Regenerate and Reformat keep them), shown one
 *  at a time under the current one: ‹ 1/2 › to step through them, each comparable with
 *  the current answer word by word, and any one made the answer again. */
export default function AnswerVersions({
  versions,
  components,
  onOpenSource,
  current,
  onUse,
}: AnswerVersionsProps) {
  const [shown, setShown] = useState<number | null>(null);
  const [comparing, setComparing] = useState(false);
  // The version and the current answer next to each other (stacked on a narrow screen):
  // a model compared with another reads better whole than as a word diff.
  const [sideBySide, setSideBySide] = useState(false);
  const Markdown = useMarkdownRenderer();
  // Opening and hiding swap the controls: focus goes to the one that takes its place.
  const toggleRef = useRef<HTMLButtonElement | null>(null);
  const hideRef = useRef<HTMLButtonElement | null>(null);
  const moveFocus = useRef(false);
  const isOpen = shown !== null;
  useEffect(() => {
    if (!moveFocus.current) return;
    moveFocus.current = false;
    (isOpen ? hideRef.current : toggleRef.current)?.focus();
  }, [isOpen]);
  const show = (index: number | null) => {
    if ((index === null) === isOpen) moveFocus.current = true;
    setShown(index);
  };
  if (versions.length === 0) return null;
  if (shown === null) {
    return (
      <button
        ref={toggleRef}
        type="button"
        className="chat-message__action chat-message__versions-toggle"
        onClick={() => show(versions.length - 1)}
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
        <span className="chat-message__versions-meta" data-testid="answer-version-meta">
          {formatTurnTime(version.timestamp)}
          {version.model ? ` · ${version.model}` : ''}
          {styleLabel(version) ? ` · ${styleLabel(version)}` : ''}
        </span>
        <button
          ref={hideRef}
          type="button"
          onClick={() => show(null)}
          aria-label="Hide earlier answers"
        >
          ✕
        </button>
      </div>
      <div className="chat-message__versions-actions">
        {current !== undefined && (
          <button
            type="button"
            className="chat-message__action"
            aria-pressed={comparing}
            onClick={() => setComparing(!comparing)}
            data-testid="answer-version-compare"
          >
            {comparing ? 'Show it as written' : 'Compare with the current answer'}
          </button>
        )}
        {current !== undefined && (
          <button
            type="button"
            className="chat-message__action"
            aria-pressed={sideBySide}
            onClick={() => {
              setSideBySide(!sideBySide);
              setComparing(false);
            }}
            data-testid="answer-version-side-by-side"
          >
            {sideBySide ? 'One at a time' : 'Side by side'}
          </button>
        )}
        {onUse && (
          <button
            type="button"
            className="chat-message__action"
            onClick={() => {
              onUse(shown);
              show(null);
            }}
            data-testid="answer-version-use"
          >
            Use this version
          </button>
        )}
      </div>
      {sideBySide && current !== undefined ? (
        <div className="chat-message__versions-side" data-testid="answer-version-side">
          <div className="chat-message__markdown" dir="auto">
            <p className="chat-message__versions-meta">This version{version.model ? ` · ${version.model}` : ''}</p>
            {Markdown ? <Markdown text={version.content} components={components} /> : <p className="chat-message__plain">{version.content}</p>}
          </div>
          <div className="chat-message__markdown" dir="auto">
            <p className="chat-message__versions-meta">The current answer</p>
            {Markdown ? <Markdown text={current} components={components} /> : <p className="chat-message__plain">{current}</p>}
          </div>
        </div>
      ) : comparing && current !== undefined ? (
        <VersionDiff before={version.content} after={current} />
      ) : (
        <div className="chat-message__markdown" dir="auto" data-testid="answer-version-content">
          {Markdown ? <Markdown text={version.content} components={components} /> : <p className="chat-message__plain">{version.content}</p>}
        </div>
      )}
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

/** The version (before) against the current answer (after): removed words struck,
 *  added ones marked. */
function VersionDiff({ before, after }: { before: string; after: string }) {
  const parts = useMemo(() => wordDiff(before, after), [before, after]);
  if (parts === null) {
    return <p className="settings__hint">These answers are too long to compare word by word.</p>;
  }
  return (
    <p className="chat-message__diff" dir="auto" data-testid="answer-version-diff">
      {parts.map((part, index) =>
        part.kind === 'same' ? (
          <span key={index}>{part.text}</span>
        ) : part.kind === 'removed' ? (
          <del key={index} className="chat-message__diff-removed">
            {part.text}
          </del>
        ) : (
          <ins key={index} className="chat-message__diff-added">
            {part.text}
          </ins>
        ),
      )}
    </p>
  );
}
