import { memo } from 'react';
import ConfirmInline from '../ConfirmInline';
import TagEditor from './TagEditor';
import { extensionOf } from './cardHelpers';
import type { DocCard } from './types';

interface DocumentCardProps {
  card: DocCard;
  disabled?: boolean;
  readOnly: boolean;
  confirming: boolean;
  deleting: boolean;
  selectable: boolean;
  selected: boolean;
  onToggleSelect: (filename: string) => void;
  onAskDelete: (key: string) => void;
  onCancelDelete: () => void;
  onConfirmDelete: (filename: string) => void;
  onRead?: (filename: string) => void;
  onSummarize?: (filenames: string[]) => void;
  onOpenOriginal?: (filename: string) => void;
  onReindex?: (filename: string) => void;
  onCancelUpload?: (uploadId: string) => void;
  onRetryUpload?: (uploadId: string) => void;
  onDismissUpload?: (uploadId: string) => void;
  onSetTags?: (filename: string, tags: string[]) => Promise<void>;
}

/** One document (or upload) card. Memoized with stable card objects and callbacks, so
 *  an upload's progress poll re-renders that card, not every card in a large corpus. */
const DocumentCard = memo(function DocumentCard({
  card,
  disabled,
  readOnly,
  confirming,
  deleting,
  selectable,
  selected,
  onToggleSelect,
  onAskDelete,
  onCancelDelete,
  onConfirmDelete,
  onRead,
  onSummarize,
  onOpenOriginal,
  onReindex,
  onCancelUpload,
  onRetryUpload,
  onDismissUpload,
  onSetTags,
}: DocumentCardProps) {
  return (
    <article
      className={`corpus-panel__card${selected ? ' corpus-panel__card--selected' : ''}`}
      data-testid="corpus-panel-item"
      data-corpus-card={card.filename}
    >
      <div className="corpus-panel__card-top">
        {selectable && (
          <input
            type="checkbox"
            className="corpus-panel__select"
            checked={selected}
            onChange={() => onToggleSelect(card.filename)}
            disabled={disabled}
            aria-label={`Select ${card.filename}`}
            data-testid="corpus-select"
          />
        )}
        <span className="corpus-panel__ext">{extensionOf(card.filename)}</span>
        <span className="corpus-panel__name">{card.filename}</span>
        {confirming ? (
          <ConfirmInline
            className="corpus-panel__confirm"
            prompt="Delete?"
            confirmLabel="Confirm"
            onCancel={onCancelDelete}
            onConfirm={() => onConfirmDelete(card.filename)}
            busy={deleting}
          />
        ) : card.status === 'indexed' ? (
          <div className="corpus-panel__card-actions">
            {onRead && (
              <button
                type="button"
                className="corpus-panel__read"
                onClick={() => onRead(card.filename)}
                aria-label={`Read ${card.filename}`}
                title="Read the indexed text"
                data-testid="corpus-read"
              >
                ☰
              </button>
            )}
            {onSummarize && (
              <button
                type="button"
                className="corpus-panel__read"
                onClick={() => onSummarize([card.filename])}
                disabled={disabled}
                aria-label={`Summarize ${card.filename}`}
                title="Summarize the whole document"
                data-testid="corpus-summarize"
              >
                ∑
              </button>
            )}
            {card.hasOriginal && onOpenOriginal && (
              <button
                type="button"
                className="corpus-panel__open-original"
                onClick={() => onOpenOriginal(card.filename)}
                aria-label={`Download the original of ${card.filename}`}
                title="Download the uploaded file"
              >
                ⇱
              </button>
            )}
            {card.canReindex && (
              <button
                type="button"
                className={`corpus-panel__reindex${card.stale ? ' corpus-panel__reindex--stale' : ''}`}
                onClick={() => onReindex?.(card.filename)}
                disabled={disabled}
                aria-label={`Re-index ${card.filename}`}
                title={
                  card.stale
                    ? 'Indexed by an older chunker — re-index from the stored original'
                    : 'Re-index from the stored original'
                }
              >
                ↻
              </button>
            )}
            {!readOnly && (
              <button
                type="button"
                className="corpus-panel__remove"
                onClick={() => onAskDelete(card.key)}
                disabled={disabled}
                aria-label={`Delete ${card.filename}`}
              >
                ✕
              </button>
            )}
          </div>
        ) : card.status === 'indexing' ? (
          !readOnly &&
          onCancelUpload &&
          card.uploadId !== undefined && (
            <button
              type="button"
              className="corpus-panel__cancel"
              onClick={() => onCancelUpload(card.uploadId!)}
              disabled={card.cancelling}
              aria-label={`Cancel indexing ${card.filename}`}
              title="Stop before anything is written to the index"
              data-testid="corpus-cancel-upload"
            >
              Cancel
            </button>
          )
        ) : (
          card.status === 'failed' &&
          card.uploadId !== undefined && (
            <div className="corpus-panel__failed-actions">
              {card.canRetry && onRetryUpload && (
                <button
                  type="button"
                  className="corpus-panel__retry"
                  onClick={() => onRetryUpload(card.uploadId!)}
                  disabled={disabled}
                  aria-label={`Retry ${card.filename}`}
                >
                  Retry
                </button>
              )}
              {onDismissUpload && (
                <button
                  type="button"
                  className="corpus-panel__remove"
                  onClick={() => onDismissUpload(card.uploadId!)}
                  aria-label={`Dismiss ${card.filename}`}
                >
                  ✕
                </button>
              )}
            </div>
          )
        )}
      </div>
      <div className="corpus-panel__status">
        <span className={`corpus-panel__dot corpus-panel__dot--${card.status}`} aria-hidden="true" />
        {card.status === 'failed' ? (
          <span role="status">{`failed — ${card.detail}`}</span>
        ) : (
          <span>{card.detail}</span>
        )}
      </div>
      {card.status === 'indexing' && (
        <div
          className="corpus-panel__bar"
          role="progressbar"
          aria-label={`Indexing ${card.filename}`}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={card.pct}
        >
          <div className="corpus-panel__bar-fill" style={{ width: `${card.pct}%` }} />
        </div>
      )}
      {card.status === 'indexed' && onSetTags && (
        <TagEditor filename={card.filename} tags={card.tags} disabled={disabled} onSave={onSetTags} />
      )}
    </article>
  );
});

export default DocumentCard;
