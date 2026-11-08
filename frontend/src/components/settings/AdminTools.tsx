import { useCallback, useEffect, useState } from 'react';
import { api, apiErrorMessage, getWorkspace } from '../../api/client';
import type {
  AnswerCacheResponse,
  AnswerFormat,
  BackupCheckResponse,
  BackupListResponse,
  ConsistencyResponse,
  WorkspaceSettingsRequest,
} from '../../api/types';
import { formatRelativeTime } from '../../utils/relativeTime';
import ConfirmInline from '../ConfirmInline';

/** The full key's tools for this workspace: its own relevance settings, its cached
 *  answers, its scheduled backups and a consistency check. Loaded when opened. */
export default function AdminTools({ onRestored }: { onRestored?: () => void }) {
  const workspace = getWorkspace() || 'default';
  return (
    <div className="settings__admin" data-testid="settings-admin">
      <WorkspaceSettings workspace={workspace} />
      <AnswerCache />
      <Backups onRestored={onRestored} />
      <Consistency />
    </div>
  );
}

const FORMATS: AnswerFormat[] = ['default', 'bullets', 'table', 'brief'];

function WorkspaceSettings({ workspace }: { workspace: string }) {
  const [draft, setDraft] = useState<WorkspaceSettingsRequest | null>(null);
  const [effective, setEffective] = useState<WorkspaceSettingsRequest | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    api
      .workspaceSettings(workspace, controller.signal)
      .then((response) => {
        setDraft(response.settings);
        setEffective(response.effective);
      })
      .catch((err) => {
        if (!controller.signal.aborted) setStatus(apiErrorMessage(err, 'Could not load them.'));
      });
    return () => controller.abort();
  }, [workspace]);

  if (draft === null || effective === null) {
    return <p className="settings__hint">{status ?? 'Loading this workspace’s settings…'}</p>;
  }
  const save = async () => {
    try {
      const response = await api.putWorkspaceSettings(workspace, draft);
      setDraft(response.settings);
      setEffective(response.effective);
      setStatus('Saved. Answers cached before were cleared.');
    } catch (err) {
      setStatus(apiErrorMessage(err, 'Could not save them.'));
    }
  };
  const number = (value: string) => {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : null;
  };
  return (
    <section aria-label="This workspace's settings" data-testid="workspace-settings">
      <h4 className="settings__subhead">Workspace “{workspace}”</h4>
      <p className="settings__hint">
        Empty fields use the server’s value (now bar {effective.rerank_min_score ?? '–'}).
      </p>
      <div className="settings__admin-grid">
        <label>
          Relevance bar
          <input
            type="number"
            min={0}
            max={1}
            step={0.01}
            value={draft.rerank_min_score ?? ''}
            onChange={(event) => setDraft({ ...draft, rerank_min_score: number(event.target.value) })}
            data-testid="workspace-settings-min-score"
          />
        </label>
        <label>
          Newer first (0–1)
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={draft.recency_boost_weight ?? ''}
            onChange={(event) =>
              setDraft({ ...draft, recency_boost_weight: number(event.target.value) })
            }
          />
        </label>
        <label>
          Answer language
          <input
            type="text"
            maxLength={40}
            placeholder="match / German"
            value={draft.answer_language ?? ''}
            onChange={(event) => setDraft({ ...draft, answer_language: event.target.value || null })}
          />
        </label>
        <label>
          Answer format
          <select
            value={draft.answer_format ?? ''}
            onChange={(event) =>
              setDraft({ ...draft, answer_format: (event.target.value as AnswerFormat) || null })
            }
          >
            <option value="">server’s</option>
            {FORMATS.map((format) => (
              <option key={format} value={format}>
                {format}
              </option>
            ))}
          </select>
        </label>
      </div>
      <button type="button" onClick={() => void save()} data-testid="workspace-settings-save">
        Save workspace settings
      </button>
      {status && (
        <p className="settings__hint" role="status">
          {status}
        </p>
      )}
    </section>
  );
}

function AnswerCache() {
  const [cache, setCache] = useState<AnswerCacheResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const load = useCallback(() => {
    api
      .answerCache()
      .then(setCache)
      .catch((err) => setError(apiErrorMessage(err, 'Could not read the cache.')));
  }, []);
  useEffect(load, [load]);
  if (error) return <p className="settings__hint">{error}</p>;
  if (cache === null) return null;
  if (!cache.enabled) {
    return <p className="settings__hint">Answers aren’t cached (ANSWER_CACHE_SIZE is 0).</p>;
  }
  return (
    <section aria-label="Cached answers" data-testid="answer-cache">
      <h4 className="settings__subhead">Cached answers</h4>
      <p className="settings__hint" data-testid="answer-cache-count">
        {cache.entries.length} in this workspace ({cache.total_entries} of {cache.max_entries} in
        all), kept {Math.round(cache.ttl_seconds / 60)} min.
      </p>
      {cache.entries.length > 0 && (
        <button
          type="button"
          onClick={() =>
            void api
              .clearAnswerCache()
              .then(load)
              .catch((err) => setError(apiErrorMessage(err, 'Could not clear the cache.')))
          }
          data-testid="answer-cache-clear"
        >
          Clear them
        </button>
      )}
    </section>
  );
}

function Backups({ onRestored }: { onRestored?: () => void }) {
  const [listing, setListing] = useState<BackupListResponse | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);
  const [checks, setChecks] = useState<Record<string, BackupCheckResponse>>({});
  const load = useCallback(() => {
    api
      .listBackups()
      .then(setListing)
      .catch((err) => setMessage(apiErrorMessage(err, 'Could not list the backups.')));
  }, []);
  useEffect(load, [load]);
  if (listing === null) {
    return message ? (
      <p className="settings__hint" data-testid="backups-off">
        {message}
      </p>
    ) : null;
  }
  return (
    <section aria-label="Scheduled backups" data-testid="backups">
      <h4 className="settings__subhead">Backups</h4>
      <p className="settings__hint">
        Every {listing.interval_hours} h, the newest {listing.keep} kept.
      </p>
      <button
        type="button"
        onClick={() =>
          void api
            .runBackup()
            .then(() => setMessage('A backup is being written; refresh the list in a moment.'))
            .catch((err) => setMessage(apiErrorMessage(err, 'Could not start a backup.')))
        }
        data-testid="backups-run"
      >
        Back up now
      </button>{' '}
      <button type="button" onClick={load} data-testid="backups-refresh">
        Refresh
      </button>
      {message && (
        <p className="settings__hint" role="status">
          {message}
        </p>
      )}
      <ul className="settings__backups">
        {listing.backups.map((backup) => {
          const check = checks[backup.name];
          const verified = check ? check.ok : backup.verified;
          return (
            <li key={backup.name} data-testid="backup-item">
              <span className="mono">{backup.name}</span> ·{' '}
              {formatRelativeTime(backup.created_at * 1000)} ·{' '}
              {verified === true ? 'verified' : verified === false ? 'damaged' : 'not checked'}{' '}
              <button
                type="button"
                onClick={() =>
                  void api
                    .verifyStoredBackup(backup.name)
                    .then((result) => setChecks((prev) => ({ ...prev, [backup.name]: result })))
                    .catch((err) => setMessage(apiErrorMessage(err, 'Could not check it.')))
                }
                data-testid="backup-verify"
              >
                Check
              </button>{' '}
              {confirming === backup.name ? (
                <ConfirmInline
                  tone="main"
                  prompt="Restore? Documents already here are kept as they are."
                  confirmLabel="Restore"
                  onCancel={() => setConfirming(null)}
                  onConfirm={() => {
                    setConfirming(null);
                    void api
                      .restoreStoredBackup(backup.name)
                      .then((outcome) => {
                        setMessage(
                          `${outcome.jobs.length} document(s) queued, ${outcome.skipped.length} already here.`,
                        );
                        onRestored?.();
                      })
                      .catch((err) => setMessage(apiErrorMessage(err, 'Could not restore it.')));
                  }}
                  confirmTestId="backup-restore-confirm"
                />
              ) : (
                <button type="button" onClick={() => setConfirming(backup.name)} data-testid="backup-restore">
                  Restore
                </button>
              )}
              {check && !check.ok && (
                <span className="settings__hint"> {check.problems.join(' ')}</span>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}

const CONSISTENCY_SECTIONS: Array<[keyof ConsistencyResponse, string]> = [
  ['originals_without_document', 'Original with no document'],
  ['documents_without_original', 'Document with no original'],
  ['versions_without_document', 'Earlier versions with no document'],
  ['incomplete', 'Incomplete (re-index repairs it)'],
];

function Consistency() {
  const [report, setReport] = useState<ConsistencyResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  return (
    <section aria-label="Consistency check" data-testid="consistency">
      <h4 className="settings__subhead">Consistency</h4>
      <button
        type="button"
        disabled={running}
        onClick={() => {
          setRunning(true);
          setError(null);
          api
            .consistency()
            .then(setReport)
            .catch((err) => setError(apiErrorMessage(err, 'Could not check.')))
            .finally(() => setRunning(false));
        }}
        data-testid="consistency-run"
      >
        {running ? 'Checking…' : 'Check the index against the originals'}
      </button>
      {error && <p className="settings__hint">{error}</p>}
      {report && (
        <div role="status" data-testid="consistency-result">
          <p className="settings__hint">
            {report.problems === 0
              ? `All ${report.documents} documents agree with their originals.`
              : `${report.problems} problem(s) among ${report.documents} documents. Nothing was changed.`}
          </p>
          <ul className="settings__backups">
            {CONSISTENCY_SECTIONS.flatMap(([key, label]) =>
              (report[key] as string[]).map((name) => (
                <li key={`${key}-${name}`} data-testid="consistency-item">
                  {label}: <span className="mono">{name}</span>
                </li>
              )),
            )}
          </ul>
        </div>
      )}
    </section>
  );
}
