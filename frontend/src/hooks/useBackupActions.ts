import { useCallback, useEffect, useRef, useState } from 'react';
import { api, apiErrorMessage } from '../api/client';
import type { DocumentJobAcceptedResponse } from '../api/types';
import { downloadFile } from '../utils/exportChat';
import { useStableHandler } from './useStableHandler';
import type { ToastInput } from './useToasts';

interface BackupOptions {
  pushToast: (toast: ToastInput) => unknown;
  noteAuthFailure: (err: unknown) => boolean;
  /** The restore queued these documents (follow them like uploads). */
  onRestoreQueued: (jobs: DocumentJobAcceptedResponse[]) => Promise<void> | void;
}

/** Corpus backups (GET /export, POST /import) and stored-original downloads, each
 *  reporting through toasts. */
export function useBackupActions({ pushToast, noteAuthFailure, onRestoreQueued }: BackupOptions) {
  const [exportingBackup, setExportingBackup] = useState(false);
  const [restoring, setRestoring] = useState(false);
  const exportingRef = useRef(false);
  const exportControllerRef = useRef<AbortController | null>(null);
  useEffect(() => () => exportControllerRef.current?.abort(), []);

  // Starts a backup download, or cancels the one running (the palette command toggles).
  const downloadBackup = useStableHandler(() => {
    if (exportingRef.current) {
      exportControllerRef.current?.abort();
      return;
    }
    exportingRef.current = true;
    setExportingBackup(true);
    const controller = new AbortController();
    exportControllerRef.current = controller;
    pushToast({ tone: 'info', title: 'Preparing backup…', body: 'The download starts when it is ready.' });
    api
      .exportCorpus(controller.signal)
      .then(({ blob, filename }) => downloadFile(filename, 'application/zip', blob))
      .catch((err) => {
        if (controller.signal.aborted) {
          pushToast({ tone: 'info', title: 'Backup cancelled' });
          return;
        }
        if (noteAuthFailure(err)) return;
        pushToast({
          tone: 'bad',
          title: 'Backup failed',
          body: apiErrorMessage(err, 'Could not build the backup.'),
        });
      })
      .finally(() => {
        exportingRef.current = false;
        exportControllerRef.current = null;
        setExportingBackup(false);
      });
  });

  const restoreBackup = useStableHandler(async (file: File) => {
    setRestoring(true);
    pushToast({ tone: 'info', title: 'Restoring backup…', body: file.name });
    let result;
    try {
      result = await api.importBackup(file);
    } catch (err) {
      if (!noteAuthFailure(err)) {
        pushToast({
          tone: 'bad',
          title: 'Restore failed',
          body: apiErrorMessage(err, 'Could not read the backup.'),
        });
      }
      return;
    } finally {
      setRestoring(false);
    }
    const notes = [
      result.skipped.length > 0 && `${result.skipped.length} already indexed (kept as is)`,
      result.deferred.length > 0 &&
        `${result.deferred.length} waiting for room in the queue — restore the same file again later`,
      result.missing_originals.length > 0 &&
        `${result.missing_originals.length} had no stored original in the backup`,
      result.rejected.length > 0 && `${result.rejected.length} couldn't be used`,
      result.feedback_imported > 0 && `${result.feedback_imported} ratings restored`,
    ].filter(Boolean);
    pushToast({
      tone: result.deferred.length + result.rejected.length > 0 ? 'warn' : 'good',
      title: `Restoring ${result.jobs.length} document${result.jobs.length === 1 ? '' : 's'}`,
      body: notes.length > 0 ? `${notes.join('; ')}.` : undefined,
    });
    if (result.jobs.length > 0) await onRestoreQueued(result.jobs);
  });

  const notify = useStableHandler(pushToast);
  const noteAuth = useStableHandler(noteAuthFailure);
  const downloadOriginal = useCallback(
    (filename: string) => {
      api
        .getDocumentOriginal(filename)
        .then((blob) => downloadFile(filename, blob.type || 'application/octet-stream', blob))
        .catch((err) => {
          if (noteAuth(err)) return;
          notify({
            tone: 'bad',
            title: 'Download failed',
            body: apiErrorMessage(err, 'Could not fetch the original.'),
          });
        });
    },
    [noteAuth, notify],
  );

  return { exportingBackup, restoring, downloadBackup, restoreBackup, downloadOriginal };
}
