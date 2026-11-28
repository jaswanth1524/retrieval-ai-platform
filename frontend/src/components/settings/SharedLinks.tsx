import { useEffect, useState } from 'react';
import { api, apiErrorMessage } from '../../api/client';
import type { ShareSummaryResponse } from '../../api/types';
import { copyText } from '../../utils/clipboard';
import { formatRelativeTimeAgo } from '../../utils/relativeTime';
import { shareLinkUrl } from '../../utils/shares';

/** Settings → Shared links: this workspace's shares made with this key, to copy a link
 *  again or delete one (its link then opens nothing). */
export default function SharedLinks() {
  const [shares, setShares] = useState<ShareSummaryResponse[] | null>(null);
  const [note, setNote] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    api.listShares(true, controller.signal).then(
      (answer) => setShares(answer.shares),
      (err: unknown) => {
        if (!controller.signal.aborted) setNote(apiErrorMessage(err, 'Could not list shares.'));
      },
    );
    return () => controller.abort();
  }, []);
  const remove = (id: string) => {
    api.deleteShare(id).then(
      () => setShares((current) => (current ?? []).filter((share) => share.id !== id)),
      (err: unknown) => setNote(apiErrorMessage(err, 'Could not delete the share.')),
    );
  };
  const copy = (id: string) => {
    void copyText(shareLinkUrl(id, window.location)).then((ok) =>
      setNote(ok ? 'Link copied.' : 'Could not copy the link.'),
    );
  };
  return (
    <div data-testid="shared-links">
      {note && (
        <p className="settings__hint" role="status">
          {note}
        </p>
      )}
      {shares === null && !note && <p className="settings__hint">Loading…</p>}
      {shares?.length === 0 && (
        <p className="settings__hint">Nothing shared from this workspace with this key yet.</p>
      )}
      {shares && shares.length > 0 && (
        <ul className="settings__list">
          {shares.map((share) => (
            <li key={share.id} data-testid="shared-link">
              <span>
                {share.title}{' '}
                <span className="settings__hint">
                  · {share.turns} answer{share.turns === 1 ? '' : 's'} ·{' '}
                  {formatRelativeTimeAgo(share.created_at * 1000)}
                </span>
              </span>{' '}
              <button type="button" onClick={() => copy(share.id)}>
                Copy link
              </button>{' '}
              <button type="button" onClick={() => remove(share.id)} data-testid="shared-link-delete">
                Delete
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
