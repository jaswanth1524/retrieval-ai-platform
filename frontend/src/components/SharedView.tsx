import { useEffect, useState } from 'react';
import { api, apiErrorMessage } from '../api/client';
import type { SharedConversationResponse } from '../api/types';
import { useDialog } from '../hooks/useDialog';
import { pageLabel } from '../utils/pages';
import { MARKDOWN_COMPONENTS } from './chat/markdownComponents';
import { useMarkdownRenderer } from './markdownLoader';
import './SummaryDialog.css';
import './SharedView.css';

interface SharedViewProps {
  shareId: string;
  onClose: () => void;
  /** Keep a copy among this browser's conversations; true when it was imported. */
  onImport?: (share: SharedConversationResponse) => boolean;
}

function when(seconds: number): string {
  return new Date(seconds * 1000).toLocaleString();
}

/** A shared answer or conversation (#s=<id>), read-only and as it was when shared —
 *  the passages' text included, so it reads the same after the documents change. */
export default function SharedView({ shareId, onClose, onImport }: SharedViewProps) {
  const dialogRef = useDialog(true, onClose);
  const Markdown = useMarkdownRenderer();
  const [share, setShare] = useState<SharedConversationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    api.readShare(shareId, controller.signal).then(setShare, (err: unknown) => {
      if (!controller.signal.aborted) {
        setError(apiErrorMessage(err, 'Could not open the shared answer.'));
      }
    });
    return () => controller.abort();
  }, [shareId]);
  return (
    <div
      className="summary-dialog__backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        className="summary-dialog shared-view"
        role="dialog"
        aria-modal="true"
        aria-label={share ? `Shared: ${share.title}` : 'Shared answer'}
        ref={dialogRef}
        data-testid="shared-view"
      >
        <h2 className="summary-dialog__title">{share ? share.title : 'Shared answer'}</h2>
        {error && (
          <p className="shared-view__error" role="alert" data-testid="shared-view-error">
            {error}
          </p>
        )}
        {!share && !error && <p className="shared-view__meta">Opening…</p>}
        {share && (
          <>
            <p className="shared-view__meta" data-testid="shared-view-meta">
              Shared by {share.created_by} on {when(share.created_at)}
              {share.workspace !== 'default' ? ` from ${share.workspace}` : ''}
              {share.expires_at ? ` · until ${when(share.expires_at)}` : ''}. As it was then:
              the documents may have changed since.
            </p>
            <ol className="shared-view__turns">
              {share.conversation.map((turn, index) => (
                <li key={index} className="shared-view__turn" data-testid="shared-view-turn">
                  {turn.question && <p className="shared-view__question">{turn.question}</p>}
                  <div className="shared-view__answer">
                    {Markdown ? (
                      <Markdown text={turn.answer} components={MARKDOWN_COMPONENTS} />
                    ) : (
                      <p>{turn.answer}</p>
                    )}
                  </div>
                  {turn.answer_status && turn.answer_status !== 'answered' && (
                    <p className="shared-view__flag">
                      {turn.answer_status === 'uncited'
                        ? 'No sources cited.'
                        : 'The documents didn’t cover this.'}
                    </p>
                  )}
                  {turn.sources.length > 0 && (
                    <details className="shared-view__sources">
                      <summary>
                        {turn.sources.length} source{turn.sources.length === 1 ? '' : 's'}
                      </summary>
                      <ol>
                        {turn.sources.map((source) => (
                          <li key={`${source.source_number}-${source.chunk_id}`} value={source.source_number}>
                            <span className="mono">
                              {source.filename} · {pageLabel(source.page, source.page_end)} ·{' '}
                              {source.section}
                            </span>
                            {source.text && <blockquote>{source.text}</blockquote>}
                          </li>
                        ))}
                      </ol>
                    </details>
                  )}
                </li>
              ))}
            </ol>
          </>
        )}
        <div className="summary-dialog__actions">
          {share && onImport && (
            <button
              type="button"
              onClick={() => {
                if (onImport(share)) onClose();
              }}
              data-testid="shared-view-import"
            >
              Import as conversation
            </button>
          )}
          <button type="button" className="summary-dialog__primary" onClick={onClose} autoFocus>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
