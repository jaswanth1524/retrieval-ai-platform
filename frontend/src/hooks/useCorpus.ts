import { useMemo, useRef, useState } from 'react';
import { ApiClientError, api } from '../api/client';
import type { DocumentJobAcceptedResponse } from '../api/types';
import type { UploadItem } from '../components/CorpusPanel';
import { newId } from '../utils/id';
import { mapWithLimit } from '../utils/pool';
import type { ToastInput } from './useToasts';

// With the server's 15 s Retry-After, about 10 minutes of waiting for a full queue.
const UPLOAD_QUEUE_MAX_ATTEMPTS = 40;
// Uploads sent at once from one batch.
const UPLOAD_CONCURRENCY = 2;

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
  const [staleFilenames, setStaleFilenames] = useState<string[]>([]);
  const [reindexableFilenames, setReindexableFilenames] = useState<string[]>([]);
  const [documentTags, setDocumentTags] = useState<Record<string, string[]>>({});
  // The full corpus currently indexed in Qdrant (across all sessions). Both the
  // corpus-management panel and the search-scope filter operate over this one list,
  // so any indexed document is scopable regardless of which session uploaded it.
  const [indexedFilenames, setIndexedFilenames] = useState<string[]>([]);
  const [chunkCounts, setChunkCounts] = useState<Record<string, number>>({});
  const [pageCounts, setPageCounts] = useState<Record<string, number>>({});
  const [byteSizes, setByteSizes] = useState<Record<string, number>>({});
  const [uploadedAts, setUploadedAts] = useState<Record<string, number>>({});
  const uploadsRef = useRef(uploads);
  uploadsRef.current = uploads;

  const actions = useMemo(() => {
    const refreshDocuments = async (signal?: AbortSignal) => {
      const documents = await api.listDocuments(signal);
      setIndexedFilenames(documents.filenames);
      // A document deleted elsewhere (another tab, the API) stayed in scope: the header
      // still named it and the popover — which lists only indexed files — couldn't
      // uncheck it.
      deps.current.pruneScopes(documents.filenames);
      setDocumentTags(documents.tags ?? {});
      setChunkCounts(documents.chunk_counts ?? {});
      setPageCounts(documents.page_counts ?? {});
      setByteSizes(documents.byte_sizes ?? {});
      setUploadedAts(documents.uploaded_ats ?? {});
      setStaleFilenames(documents.stale_filenames ?? []);
      setReindexableFilenames(documents.reindexable_filenames ?? []);
      // A key that used to be rejected now works — clear the banner.
      deps.current.onAuthRestored();
    };


    // Starts one ingest job (an upload, or a re-index of a stored original) and follows
    // it to the end, reflecting progress on the UploadItem `id`.
    // A 503 with Retry-After means the server's indexing queue is full (byte or job cap)
    // and will drain: wait and try again rather than fail every file past the cap.
    const startWhenQueueHasRoom = async (
      id: string,
      start: () => Promise<DocumentJobAcceptedResponse>,
    ): Promise<DocumentJobAcceptedResponse> => {
      for (let attempt = 1; ; attempt += 1) {
        try {
          return await start();
        } catch (err) {
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
          await new Promise((resolve) =>
            setTimeout(resolve, Math.min(Math.max(waitSeconds, 1), 30) * 1000),
          );
        }
      }
    };

    const runIngestJob = async (id: string, start: () => Promise<DocumentJobAcceptedResponse>) => {
      try {
        const accepted = await startWhenQueueHasRoom(id, start);
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
        const filename = status.result?.filename;
        if (filename) {
          // Dedupe by name — re-uploading the same filename replaces its chunks
          // server-side, so the corpus list should not grow a second entry for it.
          setIndexedFilenames((prev) => (prev.includes(filename) ? prev : [...prev, filename]));
        }
      } catch (err) {
        deps.current.onAuthFailure(err);
        setUploads((prev) =>
          prev.map((item) =>
            item.id === id
              ? { ...item, status: 'error', error: err instanceof ApiClientError ? err.message : 'Upload failed.' }
              : item,
          ),
        );
      }
    };

    const uploadOne = (file: File, id: string) => runIngestJob(id, () => api.uploadDocument(file));

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
      await runIngestJob(id, () => api.reindexDocument(filename));
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
        response.jobs.map((job, index) => runIngestJob(items[index].id, () => Promise.resolve(job))),
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
        jobs.map((job, index) => runIngestJob(items[index].id, () => Promise.resolve(job))),
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
      await runIngestJob(id, () => (file ? api.uploadDocument(file) : api.reindexDocument(filename)));
      await refreshAfterIngest();
    };

    const handleDismissUpload = (id: string) => {
      setUploads((prev) => prev.filter((item) => item.id !== id));
    };

    const handleDeleteDocument = async (filename: string) => {
      await api.deleteDocument(filename);
      // A deleted document can never remain in the corpus or the active search scope.
      setIndexedFilenames((prev) => prev.filter((name) => name !== filename));
      deps.current.removeFromScopes([filename]);
      const without = (prev: Record<string, number>) => {
        if (!(filename in prev)) return prev;
        const next = { ...prev };
        delete next[filename];
        return next;
      };
      setChunkCounts(without);
      setPageCounts(without);
      setByteSizes(without);
      setUploadedAts(without);
      setStaleFilenames((prev) => prev.filter((name) => name !== filename));
      setReindexableFilenames((prev) => prev.filter((name) => name !== filename));
      setDocumentTags((prev) => {
        if (!(filename in prev)) return prev;
        const next = { ...prev };
        delete next[filename];
        return next;
      });
    };

    const handleSetTags = async (filename: string, tags: string[]) => {
      const result = await api.setDocumentTags(filename, tags);
      setDocumentTags((prev) => {
        const next = { ...prev };
        if (result.tags.length > 0) next[filename] = result.tags;
        else delete next[filename];
        return next;
      });
    };

    return {
      refreshDocuments,
      handleUpload,
      handleRetryUpload,
      handleDismissUpload,
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
      [...new Set(Object.values(documentTags).flat())].sort((a, b) =>
        a.localeCompare(b, undefined, { sensitivity: 'base' }),
      ),
    [documentTags],
  );
  const chunkTotal = useMemo(
    () => Object.values(chunkCounts).reduce((sum, n) => sum + n, 0),
    [chunkCounts],
  );

  return {
    uploads,
    indexedFilenames,
    chunkCounts,
    pageCounts,
    byteSizes,
    uploadedAts,
    staleFilenames,
    reindexableFilenames,
    documentTags,
    availableTags,
    chunkTotal,
    ...actions,
  };
}
