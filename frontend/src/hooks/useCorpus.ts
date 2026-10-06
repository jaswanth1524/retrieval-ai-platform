import { useMemo, useRef, useState } from 'react';
import { ApiClientError, api } from '../api/client';
import type { DocumentJobAcceptedResponse, DocumentListResponse } from '../api/types';
import type { DocumentMeta, UploadItem } from '../components/CorpusPanel';
import { newId } from '../utils/id';
import { mapWithLimit } from '../utils/pool';
import type { ToastInput } from './useToasts';

// With the server's 15 s Retry-After, about 10 minutes of waiting for a full queue.
const UPLOAD_QUEUE_MAX_ATTEMPTS = 40;
// Uploads sent at once from one batch.
const UPLOAD_CONCURRENCY = 2;

/** Thrown inside an upload's own flow when Cancel stops it before the server took it. */
class UploadCancelled extends Error {}

function documentsFrom(response: DocumentListResponse): Record<string, DocumentMeta> {
  const stale = new Set(response.stale_filenames ?? []);
  const reindexable = new Set(response.reindexable_filenames ?? []);
  const documents: Record<string, DocumentMeta> = {};
  for (const filename of response.filenames) {
    documents[filename] = {
      chunks: response.chunk_counts?.[filename] ?? 0,
      pages: response.page_counts?.[filename],
      bytes: response.byte_sizes?.[filename],
      uploadedAt: response.uploaded_ats?.[filename],
      stale: stale.has(filename),
      reindexable: reindexable.has(filename),
      tags: response.tags?.[filename] ?? [],
    };
  }
  return documents;
}

export interface CorpusCallbacks {
  /** True when it handled the error (a 401), so the caller keeps its own fallback. */
  onAuthFailure: (err: unknown) => boolean;
  /** A corpus request succeeded: a previously rejected key now works. */
  onAuthRestored: () => void;
  pushToast: (toast: ToastInput) => unknown;
  pruneScopes: (indexedFilenames: string[]) => void;
  removeFromScopes: (filenames: string[]) => void;
}

/** The indexed corpus and everything that changes it: uploads and their jobs, re-index,
 *  delete and tags. Actions keep their identity across renders (they read the latest
 *  callbacks and uploads through refs), so panels receiving them can skip re-rendering. */
export function useCorpus(callbacks: CorpusCallbacks) {
  const deps = useRef(callbacks);
  deps.current = callbacks;
  const [uploads, setUploads] = useState<UploadItem[]>([]);
  // The full corpus currently indexed in Qdrant (across all sessions). Both the
  // corpus-management panel and the search-scope filter operate over this one list,
  // so any indexed document is scopable regardless of which session uploaded it.
  // Kept as its own ordered array: object keys would reorder names like "2024".
  const [indexedFilenames, setIndexedFilenames] = useState<string[]>([]);
  const [documents, setDocuments] = useState<Record<string, DocumentMeta>>({});
  const uploadsRef = useRef(uploads);
  uploadsRef.current = uploads;
  const indexedRef = useRef(indexedFilenames);
  indexedRef.current = indexedFilenames;

  const actions = useMemo(() => {
    // Per upload (by UploadItem id), outside React state so a click can't race a render:
    // the job the server accepted it as, whether Cancel was pressed before it did, and
    // how to cut short a wait for room in the queue.
    const jobIds = new Map<string, string>();
    const cancelRequested = new Set<string>();
    const wakeWaiting = new Map<string, () => void>();
    // The request still sending the file: Cancel aborts it instead of waiting for the
    // whole body to go up first.
    const sending = new Map<string, AbortController>();

    const removeUpload = (id: string) => {
      setUploads((prev) => prev.filter((item) => item.id !== id));
    };

    // Listings can come back out of order (a slow one started before a delete landing
    // after it put the deleted document back); only the latest one requested applies.
    let listingSequence = 0;

    const refreshDocuments = async (signal?: AbortSignal) => {
      const sequence = ++listingSequence;
      const response = await api.listDocuments(signal);
      if (sequence !== listingSequence) return;
      setIndexedFilenames(response.filenames);
      // A document deleted elsewhere (another tab, the API) stayed in scope: the header
      // still named it and the popover — which lists only indexed files — couldn't
      // uncheck it.
      deps.current.pruneScopes(response.filenames);
      setDocuments(documentsFrom(response));
      // A key that used to be rejected now works — clear the banner.
      deps.current.onAuthRestored();
    };


    // Starts one ingest job (an upload, or a re-index of a stored original) and follows
    // it to the end, reflecting progress on the UploadItem `id`.
    // A 503 with Retry-After means the server's indexing queue is full (byte or job cap)
    // and will drain: wait and try again rather than fail every file past the cap.
    const startWhenQueueHasRoom = async (
      id: string,
      start: (signal: AbortSignal) => Promise<DocumentJobAcceptedResponse>,
    ): Promise<DocumentJobAcceptedResponse> => {
      for (let attempt = 1; ; attempt += 1) {
        if (cancelRequested.has(id)) throw new UploadCancelled();
        const controller = new AbortController();
        sending.set(id, controller);
        try {
          return await start(controller.signal);
        } catch (err) {
          if (controller.signal.aborted && cancelRequested.has(id)) throw new UploadCancelled();
          const waitSeconds = err instanceof ApiClientError ? err.retryAfterSeconds : undefined;
          if (
            !(err instanceof ApiClientError) ||
            err.statusCode !== 503 ||
            waitSeconds === undefined ||
            attempt >= UPLOAD_QUEUE_MAX_ATTEMPTS
          ) {
            throw err;
          }
          setUploads((prev) =>
            prev.map((item) =>
              item.id === id
                ? { ...item, progress: { state: 'waiting', chunksDone: 0, chunksTotal: 0 } }
                : item,
            ),
          );
          // Cancel pressed while this attempt was being sent.
          if (cancelRequested.has(id)) throw new UploadCancelled();
          // Cancel wakes this early rather than leaving the card for up to 30 s.
          await new Promise<void>((resolve) => {
            const wake = () => {
              clearTimeout(timer);
              wakeWaiting.delete(id);
              resolve();
            };
            const timer = setTimeout(wake, Math.min(Math.max(waitSeconds, 1), 30) * 1000);
            wakeWaiting.set(id, wake);
          });
        } finally {
          sending.delete(id);
        }
      }
    };

    const requestServerCancel = async (id: string, jobId: string, filename: string) => {
      try {
        // The poll below then sees the job end as cancelled.
        await api.cancelDocumentJob(jobId);
      } catch (err) {
        // Already finished: the poll is about to report how.
        if (err instanceof ApiClientError && err.statusCode === 404) return;
        setUploads((prev) =>
          prev.map((item) => (item.id === id ? { ...item, cancelling: false } : item)),
        );
        if (deps.current.onAuthFailure(err)) return;
        deps.current.pushToast({
          tone: 'warn',
          title: `Couldn't cancel ${filename}`,
          body: err instanceof ApiClientError ? err.message : 'The server did not answer.',
        });
      }
    };

    const runIngestJob = async (
      id: string,
      filename: string,
      start: (signal: AbortSignal) => Promise<DocumentJobAcceptedResponse>,
    ) => {
      try {
        const accepted = await startWhenQueueHasRoom(id, start);
        jobIds.set(id, accepted.job_id);
        // Cancel pressed while the upload itself was still being sent.
        if (cancelRequested.has(id)) void requestServerCancel(id, accepted.job_id, filename);
        const status = await api.pollDocumentJob(accepted.job_id, (jobStatus) => {
          setUploads((prev) =>
            prev.map((item) =>
              item.id === id
                ? {
                    ...item,
                    progress: {
                      state: jobStatus.state,
                      chunksDone: jobStatus.chunks_done,
                      chunksTotal: jobStatus.chunks_total,
                    },
                  }
                : item,
            ),
          );
        });
        if (status.state === 'failed' && status.cancelled) {
          // What the user asked for; nothing was indexed, so there is nothing to show.
          removeUpload(id);
          return;
        }
        if (status.state === 'failed') {
          setUploads((prev) =>
            prev.map((item) =>
              item.id === id
                ? { ...item, status: 'error', error: status.error ?? 'Ingestion failed.' }
                : item,
            ),
          );
          return;
        }
        setUploads((prev) =>
          prev.map((item) =>
            // The file is only kept for a retry; drop it so its bytes can be freed.
            item.id === id
              ? { ...item, status: 'success', result: status.result ?? undefined, file: undefined }
              : item,
          ),
        );
        const indexedName = status.result?.filename;
        if (indexedName) {
          // Dedupe by name — re-uploading the same filename replaces its chunks
          // server-side, so the corpus list should not grow a second entry for it.
          setIndexedFilenames((prev) => (prev.includes(indexedName) ? prev : [...prev, indexedName]));
        }
      } catch (err) {
        if (err instanceof UploadCancelled) {
          removeUpload(id);
          return;
        }
        deps.current.onAuthFailure(err);
        setUploads((prev) =>
          prev.map((item) =>
            item.id === id
              ? { ...item, status: 'error', error: err instanceof ApiClientError ? err.message : 'Upload failed.' }
              : item,
          ),
        );
      } finally {
        jobIds.delete(id);
        cancelRequested.delete(id);
      }
    };

    const uploadOne = (file: File, id: string) =>
      runIngestJob(id, file.name, (signal) => api.uploadDocument(file, signal));

    // Stops an upload or re-index: before the server has it, locally; after, by asking
    // the server, which refuses (409) once the job has started writing to the index.
    const handleCancelUpload = (id: string) => {
      const item = uploadsRef.current.find((candidate) => candidate.id === id);
      if (!item || item.status !== 'uploading' || item.cancelling) return;
      setUploads((prev) =>
        prev.map((candidate) => (candidate.id === id ? { ...candidate, cancelling: true } : candidate)),
      );
      const jobId = jobIds.get(id);
      if (jobId) {
        void requestServerCancel(id, jobId, item.filename);
        return;
      }
      cancelRequested.add(id);
      sending.get(id)?.abort();
      wakeWaiting.get(id)?.();
    };

    const refreshAfterIngest = async () => {
      try {
        await refreshDocuments();
      } catch (err) {
        // Non-fatal — counts just stay stale until the next refresh.
        deps.current.onAuthFailure(err);
      }
    };

    const handleReindex = async (filename: string) => {
      const id = newId();
      setUploads((prev) => [...prev, { id, filename, status: 'uploading', reindex: true }]);
      await runIngestJob(id, filename, (signal) => api.reindexDocument(filename, signal));
      await refreshAfterIngest();
    };

    const handleReindexAllStale = async () => {
      let response;
      try {
        response = await api.reindexStaleDocuments();
      } catch (err) {
        if (!deps.current.onAuthFailure(err)) {
          deps.current.pushToast({
            tone: 'bad',
            title: 'Re-index failed',
            body: err instanceof ApiClientError ? err.message : 'Could not start re-indexing.',
          });
        }
        return;
      }
      if (response.deferred.length > 0) {
        deps.current.pushToast({
          tone: 'warn',
          title: `${response.deferred.length} document${response.deferred.length === 1 ? '' : 's'} not started`,
          body: 'The indexing queue is full. Re-index the rest once these finish.',
        });
      }
      const items: UploadItem[] = response.jobs.map((job) => ({
        id: newId(),
        filename: job.filename,
        status: 'uploading',
        reindex: true,
      }));
      setUploads((prev) => [...prev, ...items]);
      await Promise.allSettled(
        response.jobs.map((job, index) =>
          runIngestJob(items[index].id, job.filename, () => Promise.resolve(job)),
        ),
      );
      await refreshAfterIngest();
    };

    // Follows jobs the server already started (a backup restore), like uploads.
    const trackJobs = async (jobs: DocumentJobAcceptedResponse[]) => {
      const items: UploadItem[] = jobs.map((job) => ({
        id: newId(),
        filename: job.filename,
        status: 'uploading',
      }));
      setUploads((prev) => [...prev, ...items]);
      await Promise.allSettled(
        jobs.map((job, index) => runIngestJob(items[index].id, job.filename, () => Promise.resolve(job))),
      );
      await refreshAfterIngest();
    };

    const handleUpload = async (files: File[]) => {
      const newItems: UploadItem[] = files.map((file) => ({
        id: newId(),
        filename: file.name,
        status: 'uploading',
        file,
      }));
      setUploads((prev) => [...prev, ...newItems]);
      // A few at a time: each upload sends its whole file, and a 20-file drop sent 20
      // bodies at once — then all 20 waited and re-sent together when the queue was full.
      await mapWithLimit(files, UPLOAD_CONCURRENCY, (file, index) => uploadOne(file, newItems[index].id));
      // Once for the whole batch, not once per file. Chunk counts live server-side and
      // GET /documents scrolls the entire collection to compute them, so refreshing
      // inside uploadOne meant a 20-file drop triggered 20 full-collection scans.
      await refreshAfterIngest();
    };

    const handleRetryUpload = async (id: string) => {
      const item = uploadsRef.current.find((candidate) => candidate.id === id);
      if (!item || (!item.file && !item.reindex)) return;
      setUploads((prev) =>
        prev.map((candidate) =>
          candidate.id === id
            ? { ...candidate, status: 'uploading', error: undefined, progress: undefined }
            : candidate,
        ),
      );
      const { file, filename } = item;
      await runIngestJob(id, filename, (signal) =>
        file ? api.uploadDocument(file, signal) : api.reindexDocument(filename, signal),
      );
      await refreshAfterIngest();
    };

    const handleDismissUpload = (id: string) => {
      setUploads((prev) => prev.filter((item) => item.id !== id));
    };

    const handleDeleteDocument = async (filename: string) => {
      await api.deleteDocument(filename);
      // Any listing already on its way predates the delete.
      listingSequence += 1;
      // A deleted document can never remain in the corpus or the active search scope.
      setIndexedFilenames((prev) => prev.filter((name) => name !== filename));
      deps.current.removeFromScopes([filename]);
      setDocuments((prev) => {
        if (!(filename in prev)) return prev;
        const next = { ...prev };
        delete next[filename];
        return next;
      });
    };

    const handleSetTags = async (filename: string, tags: string[]) => {
      const result = await api.setDocumentTags(filename, tags);
      // Deleted while the request was out: don't bring back a record for it, whose
      // tags would linger in the scope popover.
      if (!indexedRef.current.includes(filename)) return;
      setDocuments((prev) => ({
        ...prev,
        // Indexed moments ago and not refreshed yet: the rest arrives with the next
        // listing.
        [filename]: {
          ...(prev[filename] ?? { chunks: 0, stale: false, reindexable: false }),
          tags: result.tags,
        },
      }));
    };

    return {
      refreshDocuments,
      handleUpload,
      handleRetryUpload,
      handleDismissUpload,
      handleCancelUpload,
      handleReindex,
      handleReindexAllStale,
      handleDeleteDocument,
      handleSetTags,
      trackJobs,
    };
    // Built once: everything above reads state through setters and refs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Every tag in use, for the composer's scope popover.
  const availableTags = useMemo(
    () =>
      [...new Set(Object.values(documents).flatMap((meta) => meta.tags))].sort((a, b) =>
        a.localeCompare(b, undefined, { sensitivity: 'base' }),
      ),
    [documents],
  );
  const chunkTotal = useMemo(
    () => Object.values(documents).reduce((sum, meta) => sum + meta.chunks, 0),
    [documents],
  );

  return {
    uploads,
    indexedFilenames,
    documents,
    availableTags,
    chunkTotal,
    ...actions,
  };
}
