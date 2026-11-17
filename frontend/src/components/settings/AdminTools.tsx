import { useCallback, useEffect, useRef, useState } from 'react';
import { api, apiErrorMessage, getWorkspace } from '../../api/client';
import type {
  AuditEntryResponse,
  DoctorFindingResponse,
  AnswerCacheResponse,
  AnswerFormat,
  BackupCheckResponse,
  BackupListResponse,
  ConsistencyRepairRequest,
  ConsistencyResponse,
  DocumentJobAcceptedResponse,
  WorkspaceSettingsRequest,
} from '../../api/types';
import { formatRelativeTime } from '../../utils/relativeTime';
import ConfirmInline from '../ConfirmInline';
import { useRequestSignal } from '../../hooks/useRequestSignal';
import { BACKGROUND_IMPORT_STARTED_EVENT, readZipOptions } from '../../utils/zipOptions';
import { downloadFile } from '../../utils/exportChat';

/** The full key's tools for this workspace: its own relevance settings, its cached
 *  answers, its scheduled backups and a consistency check. Loaded when opened. */
export default function AdminTools({
  onRestored,
}: {
  /** A stored backup was restored: its documents' jobs, to follow as an upload's. */
  onRestored?: (jobs: DocumentJobAcceptedResponse[]) => void;
}) {
  const workspace = getWorkspace() || 'default';
  // Bumped when the workspace's settings are saved: the server empties its cached
  // answers then, and the count below is read again.
  const [cacheVersion, setCacheVersion] = useState(0);
  return (
    <div className="settings__admin" data-testid="settings-admin">
      <WorkspaceSettings workspace={workspace} onSaved={() => setCacheVersion((n) => n + 1)} />
      <AnswerCache version={cacheVersion} />
      <Backups onRestored={onRestored} />
      <CheckUploadedBackup />
      <SettingsCheck />
      <Consistency />
      <Rebuild />
      <AuditLog />
      <NotifyTest />
      <MoveServer />
    </div>
  );
}

const FORMATS: AnswerFormat[] = ['default', 'bullets', 'table', 'brief'];

function WorkspaceSettings({ workspace, onSaved }: { workspace: string; onSaved?: () => void }) {
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
      onSaved?.();
    } catch (err) {
      setStatus(apiErrorMessage(err, 'Could not save them.'));
    }
  };
  const number = (value: string) => {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : null;
  };
  const whole = (value: string) => {
    const parsed = Number.parseInt(value, 10);
    return Number.isInteger(parsed) ? parsed : null;
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
        <label>
          Closest passages when none clears the bar (0 = say not found)
          <input
            type="number"
            min={0}
            max={20}
            placeholder={String(effective.rerank_fallback_count ?? '')}
            value={draft.rerank_fallback_count ?? ''}
            onChange={(event) =>
              setDraft({ ...draft, rerank_fallback_count: whole(event.target.value) })
            }
            data-testid="workspace-settings-fallback"
          />
        </label>
        <label>
          Relative floor (0–1)
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            placeholder={String(effective.rerank_relative_floor ?? '')}
            value={draft.rerank_relative_floor ?? ''}
            onChange={(event) =>
              setDraft({ ...draft, rerank_relative_floor: number(event.target.value) })
            }
          />
        </label>
        <label>
          Passages per answer
          <input
            type="number"
            min={1}
            max={50}
            placeholder={String(effective.max_context_chunks ?? '')}
            value={draft.max_context_chunks ?? ''}
            onChange={(event) => setDraft({ ...draft, max_context_chunks: whole(event.target.value) })}
            data-testid="workspace-settings-passages"
          />
        </label>
        <label>
          Suggested follow-ups
          <select
            value={draft.followups_enabled == null ? '' : draft.followups_enabled ? 'on' : 'off'}
            onChange={(event) =>
              setDraft({
                ...draft,
                followups_enabled: event.target.value === '' ? null : event.target.value === 'on',
              })
            }
          >
            <option value="">server’s</option>
            <option value="on">on</option>
            <option value="off">off</option>
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

const BACKUP_POLL_MS = 5000;
const BACKUP_POLL_TRIES = 12;

// Backup names carry their time (docrag-YYYYmmdd-HHMMSS.zip): the newest sorts last.
const newest = (response: BackupListResponse) =>
  response.backups.map((backup) => backup.name).sort().at(-1) ?? null;

function AnswerCache({ version = 0 }: { version?: number }) {
  const [cache, setCache] = useState<AnswerCacheResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [clearing, setClearing] = useState(false);
  // Settings closed before a read came back: its answer goes nowhere.
  const alive = useRef(true);
  useEffect(
    () => () => {
      alive.current = false;
    },
    [],
  );
  const load = useCallback(() => {
    api
      .answerCache()
      .then((response) => {
        if (!alive.current) return;
        // A read that works clears an earlier failure (it stayed until reopened).
        setError(null);
        setCache(response);
      })
      .catch((err) => alive.current && setError(apiErrorMessage(err, 'Could not read the cache.')));
  }, []);
  useEffect(load, [load, version]);
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
        all), kept {Math.round(cache.ttl_seconds / 60)} min
        {cache.in_flight ? `; ${cache.in_flight} being answered now` : ''}.
      </p>
      {cache.entries.length > 0 && (
        <ul className="settings__backups" data-testid="answer-cache-entries">
          {cache.entries.slice(0, 20).map((entry, index) => (
            <li key={`${index}-${entry.question}`} data-testid="answer-cache-entry">
              <span dir="auto">{entry.question}</span> ·{' '}
              <span className="mono">{formatAge(entry.age_seconds)}</span>
            </li>
          ))}
          {cache.entries.length > 20 && <li>and {cache.entries.length - 20} more</li>}
        </ul>
      )}
      {cache.entries.length > 0 &&
        (clearing ? (
          <ConfirmInline
            tone="main"
                  prompt="Clear every cached answer in this workspace?"
            confirmLabel="Clear"
            onCancel={() => setClearing(false)}
            onConfirm={() => {
              setClearing(false);
              void api
                .clearAnswerCache()
                .then(load)
                .catch(
                  (err) => alive.current && setError(apiErrorMessage(err, 'Could not clear the cache.')),
                );
            }}
            confirmTestId="answer-cache-clear-confirm"
          />
        ) : (
          <button type="button" onClick={() => setClearing(true)} data-testid="answer-cache-clear">
            Clear them
          </button>
        ))}
    </section>
  );
}

function Backups({
  onRestored,
}: {
  onRestored?: (jobs: DocumentJobAcceptedResponse[]) => void;
}) {
  const [listing, setListing] = useState<BackupListResponse | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [checks, setChecks] = useState<Record<string, BackupCheckResponse>>({});
  // The stored backup being checked (one at a time: the server checks one at a time).
  const [checking, setChecking] = useState<string | null>(null);
  // A restore in the background queues every document as room frees (followed under
  // Imports in the corpus panel); now, it queues the queue's worth and defers the rest.
  const [inBackground, setInBackground] = useState(() => readZipOptions().background);
  const checkRequest = useRequestSignal();
  const alive = useRef(true);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(
    () => () => {
      alive.current = false;
      if (pollTimer.current !== null) clearTimeout(pollTimer.current);
    },
    [],
  );
  const load = useCallback(() => {
    api
      .listBackups()
      .then((response) => alive.current && setListing(response))
      .catch(
        (err) => alive.current && setMessage(apiErrorMessage(err, 'Could not list the backups.')),
      );
  }, []);
  useEffect(load, [load]);
  // After "Back up now": the list is read again until the new backup shows (or a minute
  // passes), instead of asking the reader to refresh it.
  const followNewBackup = useCallback((known: string | null, triesLeft: number) => {
    pollTimer.current = setTimeout(() => {
      void api
        .listBackups()
        .then((response) => {
          if (!alive.current) return;
          setListing(response);
          if (newest(response) !== known) {
            setMessage('The backup is written.');
          } else if (triesLeft > 1) {
            followNewBackup(known, triesLeft - 1);
          }
        })
        .catch(() => undefined);
    }, BACKUP_POLL_MS);
  }, []);
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
            .then(() => {
              setMessage('A backup is being written…');
              followNewBackup(newest(listing), BACKUP_POLL_TRIES);
            })
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
      <label className="settings__row">
        <input
          type="checkbox"
          checked={inBackground}
          onChange={(event) => setInBackground(event.target.checked)}
          data-testid="backups-restore-background"
        />{' '}
        Restore in the background (every document queued as room frees, instead of the
        queue&apos;s worth now and the rest deferred)
      </label>
      <ul className="settings__backups">
        {listing.backups.map((backup) => {
          const check = checks[backup.name];
          const verified = check ? check.ok : backup.verified;
          return (
            <li key={backup.name} data-testid="backup-item">
              <span className="mono">{backup.name}</span> ·{' '}
              {formatRelativeTime(backup.created_at * 1000)} ·{' '}
              {verified === true ? 'verified' : verified === false ? 'damaged' : 'not checked'}{' '}
              {checking === backup.name ? (
                <button
                  type="button"
                  onClick={() => {
                    checkRequest.cancel();
                    setChecking(null);
                    setMessage('Check cancelled.');
                  }}
                  data-testid="backup-verify-cancel"
                >
                  Checking… Cancel
                </button>
              ) : (
                <button
                  type="button"
                  disabled={checking !== null}
                  onClick={() => {
                    const signal = checkRequest.start();
                    setChecking(backup.name);
                    void api
                      .verifyStoredBackup(backup.name, signal)
                      .then((result) => setChecks((prev) => ({ ...prev, [backup.name]: result })))
                      .catch((err) => {
                        if (!signal.aborted) setMessage(apiErrorMessage(err, 'Could not check it.'));
                      })
                      .finally(() => {
                        if (!signal.aborted) setChecking(null);
                      });
                  }}
                  data-testid="backup-verify"
                >
                  Check
                </button>
              )}{' '}
              <button
                type="button"
                onClick={() =>
                  void api
                    .downloadStoredBackup(backup.name)
                    .then((blob) => downloadFile(backup.name, 'application/zip', blob))
                    .catch((err) => setMessage(apiErrorMessage(err, 'Could not download it.')))
                }
                data-testid="backup-download"
              >
                Download
              </button>{' '}
              {deleting === backup.name ? (
                <ConfirmInline
                  tone="main"
                  prompt="Delete this backup?"
                  confirmLabel="Delete"
                  onCancel={() => setDeleting(null)}
                  onConfirm={() => {
                    setDeleting(null);
                    void api
                      .deleteStoredBackup(backup.name)
                      .then(() => {
                        setMessage(`Deleted ${backup.name}.`);
                        load();
                      })
                      .catch((err) => setMessage(apiErrorMessage(err, 'Could not delete it.')));
                  }}
                  confirmTestId="backup-delete-confirm"
                />
              ) : (
                <button type="button" onClick={() => setDeleting(backup.name)} data-testid="backup-delete">
                  Delete
                </button>
              )}{' '}
              {confirming === backup.name ? (
                <ConfirmInline
                  tone="main"
                  prompt="Restore? Documents already here are kept as they are."
                  confirmLabel="Restore"
                  onCancel={() => setConfirming(null)}
                  onConfirm={() => {
                    setConfirming(null);
                    void api
                      .restoreStoredBackup(backup.name, inBackground)
                      .then((outcome) => {
                        if (outcome.import_id) {
                          window.dispatchEvent(new Event(BACKGROUND_IMPORT_STARTED_EVENT));
                          setMessage(
                            `Restoring ${backup.name} in the background: its documents are queued as there is room. Follow it under Imports in the corpus panel.`,
                          );
                          return;
                        }
                        // What didn't come back is said too: a partial restore read
                        // as a whole one.
                        const left = [
                          outcome.deferred?.length
                            ? `${outcome.deferred.length} not queued (the indexing queue was full: restore again later)`
                            : '',
                          outcome.rejected?.length ? `${outcome.rejected.length} refused` : '',
                          outcome.missing_originals?.length
                            ? `${outcome.missing_originals.length} with no original in the backup`
                            : '',
                        ].filter(Boolean);
                        setMessage(
                          `${outcome.jobs.length} document(s) queued, ${outcome.skipped.length} already here` +
                            (left.length ? `; ${left.join(', ')}.` : '.'),
                        );
                        onRestored?.(outcome.jobs);
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
              {check && checkProblems(check) && (
                <span className="settings__hint"> {checkProblems(check)}</span>
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

// Remedies the repair offers, by the finding they put right.
const REMEDIES: Array<[keyof ConsistencyResponse, keyof ConsistencyRepairRequest, string]> = [
  ['originals_without_document', 'delete_orphan_originals', 'Delete the originals with no document'],
  ['versions_without_document', 'delete_orphan_versions', 'Delete the earlier versions with no document'],
  ['incomplete', 'reindex_incomplete', 'Re-index the incomplete documents'],
];

function SettingsCheck() {
  const [findings, setFindings] = useState<DoctorFindingResponse[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const run = () => {
    setRunning(true);
    setError(null);
    api
      .doctor()
      .then((report) => setFindings(report.findings))
      .catch((err) => setError(apiErrorMessage(err, 'Could not check.')))
      .finally(() => setRunning(false));
  };
  return (
    <section aria-label="Settings check" data-testid="doctor">
      <h4 className="settings__subhead">Settings check</h4>
      <p className="settings__hint">
        Backups, free disk, the memory budget and access, as this server is set up.
      </p>
      <button type="button" onClick={run} disabled={running} data-testid="doctor-run">
        {running ? 'Checking…' : 'Check settings'}
      </button>
      {error && (
        <p className="settings__hint" role="alert">
          {error}
        </p>
      )}
      {findings !== null && (
        <div role="status" data-testid="doctor-result">
          {findings.length === 0 ? (
            <p className="settings__hint">Nothing to report.</p>
          ) : (
            <ul className="settings__backups">
              {findings.map((finding) => (
                <li key={`${finding.subject}:${finding.message}`} data-testid="doctor-finding">
                  <strong>{finding.level === 'warning' ? 'Warning' : 'Note'}</strong>{' '}
                  <span className="mono">{finding.subject}</span>: {finding.message}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}

function Consistency() {
  const [report, setReport] = useState<ConsistencyResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [chosen, setChosen] = useState<ConsistencyRepairRequest>({});
  const [confirming, setConfirming] = useState(false);
  const [repaired, setRepaired] = useState<string | null>(null);
  const check = () => {
    setRunning(true);
    setError(null);
    api
      .consistency()
      .then((next) => {
        setReport(next);
        setChosen({});
      })
      .catch((err) => setError(apiErrorMessage(err, 'Could not check.')))
      .finally(() => setRunning(false));
  };
  const offered = report
    ? REMEDIES.filter(([finding]) => (report[finding] as string[]).length > 0)
    : [];
  const repair = () => {
    setConfirming(false);
    api
      .repairConsistency(chosen)
      .then((outcome) => {
        const done =
          outcome.originals_deleted.length + outcome.versions_deleted.length + outcome.reindex_jobs.length;
        setRepaired(
          `${done} put right${outcome.left.length ? `; ${outcome.left.length} left as they were` : ''}.`,
        );
        check();
      })
      .catch((err) => setRepaired(apiErrorMessage(err, 'Could not repair.')));
  };
  return (
    <section aria-label="Consistency check" data-testid="consistency">
      <h4 className="settings__subhead">Consistency</h4>
      <button type="button" disabled={running} onClick={check} data-testid="consistency-run">
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
          {offered.length > 0 && (
            <fieldset className="settings__repair" data-testid="consistency-repair">
              <legend>Put right</legend>
              {offered.map(([, remedy, label]) => (
                <label key={remedy}>
                  <input
                    type="checkbox"
                    checked={chosen[remedy] === true}
                    onChange={(event) => setChosen({ ...chosen, [remedy]: event.target.checked })}
                    data-testid={`consistency-repair-${remedy}`}
                  />{' '}
                  {label}
                </label>
              ))}
              {confirming ? (
                <ConfirmInline
                  tone="main"
                  prompt="Deleted originals can't be brought back. Go ahead?"
                  confirmLabel="Repair"
                  onCancel={() => setConfirming(false)}
                  onConfirm={repair}
                  confirmTestId="consistency-repair-confirm"
                />
              ) : (
                <button
                  type="button"
                  disabled={!Object.values(chosen).some(Boolean)}
                  onClick={() => setConfirming(true)}
                  data-testid="consistency-repair-run"
                >
                  Repair
                </button>
              )}
            </fieldset>
          )}
        </div>
      )}
      {repaired && (
        <p className="settings__hint" role="status" data-testid="consistency-repaired">
          {repaired}
        </p>
      )}
    </section>
  );
}

/** "40 s ago", "12 min ago". */
function formatAge(seconds: number): string {
  if (seconds < 60) return `${Math.max(0, Math.round(seconds))} s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)} min ago`;
  return `${Math.round(seconds / 3600)} h ago`;
}

/** What a check found: damage, then documents a restore would refuse (not damage: the
 *  backup is intact, but they wouldn't come back). */
function checkProblems(check: BackupCheckResponse): string {
  const refused = check.rejected ?? [];
  return [
    ...check.problems,
    ...(refused.length > 0 ? [`a restore would refuse ${refused.length}: ${refused.join('; ')}`] : []),
  ].join(' · ');
}

/** Any backup zip read through before restoring it (Download a backup's, or one kept
 *  elsewhere): its checksums, and what a restore would bring back or refuse. */
function CheckUploadedBackup() {
  const [result, setResult] = useState<{ name: string; check: BackupCheckResponse } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const checkRequest = useRequestSignal();
  return (
    <section aria-label="Check a backup file" data-testid="backup-check-upload">
      <h4 className="settings__subhead">Check a backup file</h4>
      <input
        type="file"
        accept=".zip,application/zip"
        aria-label="Backup zip to check"
        disabled={busy}
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = '';
          if (!file) return;
          setBusy(true);
          setError(null);
          const signal = checkRequest.start();
          api
            .verifyBackup(file, signal)
            .then((check) => setResult({ name: file.name, check }))
            .catch((err) => {
              if (!signal.aborted) setError(apiErrorMessage(err, 'Could not check it.'));
            })
            .finally(() => {
              if (!signal.aborted) setBusy(false);
            });
        }}
        data-testid="backup-check-file"
      />
      {busy && (
        <p className="settings__hint">
          Reading it…{' '}
          <button
            type="button"
            onClick={() => {
              checkRequest.cancel();
              setBusy(false);
            }}
            data-testid="backup-check-cancel"
          >
            Cancel
          </button>
        </p>
      )}
      {error && <p className="settings__hint">{error}</p>}
      {result && (
        <p className="settings__hint" role="status" data-testid="backup-check-result">
          {result.name}: {result.check.documents} document(s)
          {typeof result.check.would_restore === 'number'
            ? `, ${result.check.would_restore} a restore would bring back`
            : ''}
          {checkProblems(result.check) ? ` — ${checkProblems(result.check)}` : ' — restores as written.'}
        </p>
      )}
    </section>
  );
}


/** Index every stored original with no document again: after a lost index. */
function Rebuild() {
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const run = () => {
    setConfirming(false);
    setRunning(true);
    setError(null);
    api
      .rebuild()
      .then((started) =>
        setResult(
          started.import_id
            ? `Indexing ${started.originals} document${started.originals === 1 ? '' : 's'} from their originals — the corpus panel’s background imports follow it.`
            : 'Nothing to rebuild: every stored original has its document.',
        ),
      )
      .catch((err) => setError(apiErrorMessage(err, 'Could not start the rebuild.')))
      .finally(() => setRunning(false));
  };
  return (
    <section aria-label="Rebuild from originals" data-testid="rebuild">
      <h4 className="settings__subhead">Rebuild from originals</h4>
      <p className="settings__hint">
        After a lost index (a wiped Qdrant volume, a move): every kept original with no
        document is indexed again, with its tags and source.
      </p>
      {confirming ? (
        <ConfirmInline
          prompt="Index every original that has no document?"
          confirmLabel="Rebuild"
          tone="main"
          onConfirm={run}
          onCancel={() => setConfirming(false)}
        />
      ) : (
        <button type="button" onClick={() => setConfirming(true)} disabled={running} data-testid="rebuild-run">
          {running ? 'Starting…' : 'Rebuild'}
        </button>
      )}
      {error && (
        <p className="settings__hint" role="alert">
          {error}
        </p>
      )}
      {result && (
        <p className="settings__hint" role="status" data-testid="rebuild-result">
          {result}
        </p>
      )}
    </section>
  );
}

/** The audit log (AUDIT_LOG_PATH): who changed what, newest first. */
function AuditLog() {
  const [entries, setEntries] = useState<AuditEntryResponse[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [event, setEvent] = useState('');
  const [loading, setLoading] = useState(false);
  const load = () => {
    setLoading(true);
    setError(null);
    api
      .audit({ event: event.trim() || undefined, limit: 200 })
      .then((report) => setEntries(report.entries))
      .catch((err) => setError(apiErrorMessage(err, 'Could not read the audit log.')))
      .finally(() => setLoading(false));
  };
  return (
    <section aria-label="Audit log" data-testid="audit-log">
      <h4 className="settings__subhead">Audit log</h4>
      <p className="settings__hint">
        Uploads, deletes, tag changes, restores and settings, with the kind of key used.
      </p>
      <form
        className="settings__row"
        onSubmit={(formEvent) => {
          formEvent.preventDefault();
          load();
        }}
      >
        <input
          className="settings__input"
          value={event}
          onChange={(change) => setEvent(change.target.value)}
          placeholder="Event, or a prefix like document."
          aria-label="Only this event (or prefix ending in a dot)"
          data-testid="audit-event"
        />
        <button type="submit" disabled={loading} data-testid="audit-load">
          {loading ? 'Loading…' : entries ? 'Reload' : 'Show'}
        </button>
      </form>
      {error && (
        <p className="settings__hint" role="alert">
          {error}
        </p>
      )}
      {entries !== null &&
        (entries.length === 0 ? (
          <p className="settings__hint">No events.</p>
        ) : (
          <ul className="settings__backups" data-testid="audit-entries">
            {entries.map((entry, index) => (
              <li key={`${entry.created_at}:${index}`} data-testid="audit-entry">
                <span className="mono">{formatRelativeTime(entry.created_at * 1000)}</span>{' '}
                <strong>{entry.event}</strong> <span className="mono">({entry.key})</span>{' '}
                {auditDetail(entry)}
              </li>
            ))}
          </ul>
        ))}
    </section>
  );
}

/** An audit event's fields in one line ("filename=guide.pdf, tags=hr"). */
function auditDetail(entry: AuditEntryResponse): string {
  return Object.entries(entry.fields ?? {})
    .filter(([, value]) => value !== null && value !== undefined && value !== '')
    .slice(0, 6)
    .map(([name, value]) => `${name}=${Array.isArray(value) ? value.join(' ') : String(value)}`)
    .join(', ')
    .slice(0, 300);
}

/** Send one notification through NOTIFY_URL, to check it reaches the operator. */
function NotifyTest() {
  const [result, setResult] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const run = () => {
    setRunning(true);
    api
      .notifyTest()
      .then((sent) =>
        setResult(sent.sent ? 'Sent — check where notifications arrive.' : `Not sent: ${sent.error ?? 'no reason given'}`),
      )
      .catch((err) => setResult(apiErrorMessage(err, 'Notifications are off (NOTIFY_URL).')))
      .finally(() => setRunning(false));
  };
  return (
    <section aria-label="Failure notifications" data-testid="notify-test">
      <h4 className="settings__subhead">Failure notifications</h4>
      <p className="settings__hint">
        A failed backup, a file the watcher gave up on or a failed scheduled re-index is
        posted to NOTIFY_URL.
      </p>
      <button type="button" onClick={run} disabled={running} data-testid="notify-test-run">
        {running ? 'Sending…' : 'Send a test notification'}
      </button>
      {result && (
        <p className="settings__hint" role="status" data-testid="notify-test-result">
          {result}
        </p>
      )}
    </section>
  );
}

/** The whole server as one file (every workspace, with index snapshots), and back. */
function MoveServer() {
  const [busy, setBusy] = useState<'pack' | 'unpack' | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const pack = () => {
    setBusy('pack');
    setMessage(null);
    api
      .migrateExport()
      .then(({ blob, filename }) => downloadFile(filename, 'application/zip', blob))
      .catch((err) => setMessage(apiErrorMessage(err, 'Could not write the bundle.')))
      .finally(() => setBusy(null));
  };
  const unpack = (file: File) => {
    setBusy('unpack');
    setMessage(null);
    api
      .migrateImport(file)
      .then((started) =>
        setMessage(
          `Unpacking ${started.workspaces.join(', ') || 'nothing'} — the default workspace’s background imports follow it.`,
        ),
      )
      .catch((err) => setMessage(apiErrorMessage(err, 'Could not unpack the bundle.')))
      .finally(() => setBusy(null));
  };
  return (
    <section aria-label="Move this server" data-testid="move-server">
      <h4 className="settings__subhead">Move this server</h4>
      <p className="settings__hint">
        One file with every workspace’s documents, history, settings and index. Unpacked on
        a server with the same embedding setup, nothing is embedded again.
      </p>
      <div className="settings__row">
        <button type="button" onClick={pack} disabled={busy !== null} data-testid="migrate-pack">
          {busy === 'pack' ? 'Writing…' : 'Download bundle'}
        </button>
        <button
          type="button"
          onClick={() => fileRef.current?.click()}
          disabled={busy !== null}
          data-testid="migrate-unpack"
        >
          {busy === 'unpack' ? 'Uploading…' : 'Unpack a bundle…'}
        </button>
        <input
          ref={fileRef}
          type="file"
          accept=".zip,application/zip"
          hidden
          onChange={(change) => {
            const file = change.target.files?.[0];
            change.target.value = '';
            if (file) unpack(file);
          }}
          data-testid="migrate-file"
        />
      </div>
      {message && (
        <p className="settings__hint" role="status" data-testid="migrate-result">
          {message}
        </p>
      )}
    </section>
  );
}
