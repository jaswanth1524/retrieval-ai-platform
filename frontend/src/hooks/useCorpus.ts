import { useMemo, useRef, useState } from 'react';
import { ApiClientError, api, apiErrorMessage, getWorkspace } from '../api/client';
import type { DocumentJobAcceptedResponse, DocumentListResponse } from '../api/types';
import type { DocumentMeta, UploadItem } from '../components/CorpusPanel';
import { newId } from '../utils/id';
import { createLimiter } from '../utils/pool';
import type { ToastInput } from './useToasts';

// With the server's 15 s Retry-After, about 10 minutes of waiting for a full queue.
const UPLOAD_QUEUE_MAX_ATTEMPTS = 40;
// Upload bodies sent at once. Only the sending is limited: once the server has accepted
// a file, its job is followed on its own, so the next file goes up while earlier ones
// are still being indexed (the server's queue, 503 + Retry-After, does the admitting).
const UPLOAD_CONCURRENCY = 2;

/** Thrown inside an upload's own flow when Cancel stops it before the server took it. */
class UploadCancelled extends Error {}

function documentsFrom(response: DocumentListResponse): Record<string, DocumentMeta> {
  const stale = new Set(response.stale_filenames ?? []);
  const incomplete = new Set(response.incomplete_filenames ?? []);
  const reindexable = new Set(response.reindexable_filenames ?? []);
  const documents: Record<string, DocumentMeta> = {};
  for (const filename of response.filenames) {
    documents[filename] = {
      chunks: response.chunk_counts?.[filename] ?? 0,
      pages: response.page_counts?.[filename],
      bytes: response.byte_sizes?.[filename],
      uploadedAt: response.uploaded_ats?.[filename],
      stale: stale.has(filename),
      ...(incomplete.has(filename) ? { incomplete: true } : {}),
      reindexable: reindexable.has(filename),
      tags: response.tags?.[filename] ?? [],
      ...(response.source_urls?.[filename] ? { sourceUrl: response.source_urls[filename] } : {}),
      ...(response.url_refresh?.[filename] ? { urlRefresh: response.url_refresh[filename] } : {}),
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
  pruneScopes: (indexedFilenames: string[], indexedTags?: string[]) => void;
  /** `workspace`: the one the documents were removed from (its conversations' scopes). */
  removeFromScopes: (filenames: string[], workspace?: string) => void;
}

/** The indexed corpus and everything that changes it: uploads and their jobs, re-index,
 *  delete and tags. Actions keep their identity across renders (they read the latest
 *  callbacks and uploads through refs), so panels receiving them can skip re-rendering.
 *
 *  The corpus is the workspace in use's (api/client getWorkspace). Every change goes to
 *  the workspace it was started in — an upload waiting for room in the queue, a retry, a
 *  bulk delete — even after a switch, and one that ends in a workspace no longer in use
 *  leaves the lists here (now another workspace's) alone. */
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
  // Tags given to every document this tab uploads or adds from a URL until changed.
  const [uploadTags, setUploadTags] = useState<string[]>([]);
  const uploadTagsRef = useRef(uploadTags);
  uploadTagsRef.current = uploadTags;
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
    // Shared by every batch: two drops in a row still send at most two files at once.
    const sendSlots = createLimiter(UPLOAD_CONCURRENCY);

    const removeUpload = (id: string) => {
      setUploads((prev) => prev.filter((item) => item.id !== id));
    };

    const isCurrent = (workspace: string) => workspace === getWorkspace();

    // Listings can come back out of order (a slow one started before a delete landing
    // after it put the deleted document back); only the latest one requested applies.
    let listingSequence = 0;

    const refreshDocuments = async (signal?: AbortSignal) => {
      const workspace = getWorkspace();
      const sequence = ++listingSequence;
      const response = await api.listDocuments(signal, workspace);
      // Another workspace's listing, from before a switch.
      if (sequence !== listingSequence || !isCurrent(workspace)) return;
      setIndexedFilenames(response.filenames);
      // A document deleted elsewhere (another tab, the API) stayed in scope: the header
      // still named it and the popover — which lists only indexed files — couldn't
      // uncheck it.
      // Tags only when the server reports them (an older one doesn't), or every tag
      // in scope would be dropped.
      deps.current.pruneScopes(
        response.filenames,
        response.tags ? Object.values(response.tags).flat() : undefined,
      );
      setDocuments(documentsFrom(response));
      // A key that used to be rejected now works — clear the banner.
      deps.current.onAuthRestored();
    };

    // Another workspace is in use: nothing listed so far is its, and a card left from the
    // previous one could act on it (its documents' names may exist here too). Failed and
    // in-flight uploads stay, shown only in their own workspace (App filters them).
    const resetForWorkspace = () => {
      listingSequence += 1;
      setIndexedFilenames([]);
      setDocuments({});
      // Tags typed for the previous workspace's uploads would tag this one's.
      setUploadTags([]);
      setUploads((prev) =>
        prev.some((item) => item.status === 'success')
          ? prev.filter((item) => item.status !== 'success')
          : prev,
      );
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

    const requestServerCancel = async (
      id: string,
      jobId: string,
      filename: string,
      workspace: string,
    ) => {
      try {
        // The poll below then sees the job end as cancelled.
        await api.cancelDocumentJob(jobId, undefined, workspace);
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
          body: apiErrorMessage(err, 'The server did not answer.'),
        });
      }
    };

    // `workspace`: the one the job was started in (its UploadItem's); `start` sends there.
    const runIngestJob = async (
      id: string,
      filename: string,
      workspace: string,
      start: (signal: AbortSignal) => Promise<DocumentJobAcceptedResponse>,
    ) => {
      try {
        const accepted = await startWhenQueueHasRoom(id, start);
        jobIds.set(id, accepted.job_id);
        // Cancel pressed while the upload itself was still being sent.
        if (cancelRequested.has(id)) void requestServerCancel(id, accepted.job_id, filename, workspace);
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
        }, undefined, workspace);
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
        // Indexed in a workspace no longer in use: the list here is another one's.
        if (indexedName && isCurrent(workspace)) {
          // Dedupe by name — re-uploading the same filename replaces its chunks
          // server-side, so the corpus list should not grow a second entry for it.
          setIndexedFilenames((prev) => (prev.includes(indexedName) ? prev : [...prev, indexedName]));
          // The listing is refreshed once the whole batch ends; until then a new card
          // read "0 chunks". The job's own count is shown in the meantime.
          const chunks = status.result?.chunks_ingested;
          if (typeof chunks === 'number') {
            setDocuments((prev) => {
              const before = prev[indexedName] as DocumentMeta | undefined;
              return {
                ...prev,
                [indexedName]: before
                  ? { ...before, chunks }
                  : { chunks, stale: false, reindexable: false, tags: [] },
              };
            });
          }
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
              ? { ...item, status: 'error', error: apiErrorMessage(err, 'Upload failed.') }
              : item,
          ),
        );
      } finally {
        jobIds.delete(id);
        cancelRequested.delete(id);
      }
    };

    const currentUploadTags = () =>
      uploadTagsRef.current.length > 0 ? [...uploadTagsRef.current] : undefined;

    // Every upload body, first try or retry, goes through the same few send slots.
    const sendUpload = (file: File, signal: AbortSignal, workspace: string, tags?: string[]) =>
      sendSlots.run(() => api.uploadDocument(file, signal, workspace, tags), signal);

    const uploadOne = (file: File, id: string, workspace: string, tags?: string[]) =>
      runIngestJob(id, file.name, workspace, (signal) =>
        sendUpload(file, signal, workspace, tags),
      );

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
        void requestServerCancel(id, jobId, item.filename, item.workspace ?? '');
        return;
      }
      cancelRequested.add(id);
      sending.get(id)?.abort();
      wakeWaiting.get(id)?.();
    };

    const refreshAfterIngest = async (workspace: string) => {
      // Finished in a workspace no longer in use: its listing is fetched on the way back.
      if (!isCurrent(workspace)) {
        setUploads((prev) =>
          prev.filter((item) => item.status !== 'success' || item.workspace !== workspace),
        );
        return;
      }
      try {
        await refreshDocuments();
        // Finished cards aren't shown (the document's own card is), so once the listing
        // has them, drop them: kept, they piled up for the whole session.
        setUploads((prev) =>
          prev.some((item) => item.status === 'success')
            ? prev.filter((item) => item.status !== 'success')
            : prev,
        );
      } catch (err) {
        // Non-fatal — counts just stay stale until the next refresh.
        deps.current.onAuthFailure(err);
      }
    };

    // The server fetches the document; the card shows the URL until it names the file.
    const startFromUrl =
      (id: string, url: string, workspace: string, tags?: string[]) =>
      async (signal: AbortSignal) => {
        const accepted = await api.uploadFromUrl(url, signal, workspace, tags);
        setUploads((prev) =>
          prev.map((item) => (item.id === id ? { ...item, filename: accepted.filename } : item)),
        );
        return accepted;
      };

    const handleUploadFromUrl = async (url: string) => {
      const id = newId();
      const workspace = getWorkspace();
      const tags = currentUploadTags();
      setUploads((prev) => [
        ...prev,
        { id, filename: url, status: 'uploading', sourceUrl: url, workspace, tags },
      ]);
      await runIngestJob(id, url, workspace, startFromUrl(id, url, workspace, tags));
      await refreshAfterIngest(workspace);
    };

    const handleReindex = async (filename: string, workspace = getWorkspace()) => {
      const id = newId();
      setUploads((prev) => [...prev, { id, filename, status: 'uploading', reindex: true, workspace }]);
      await runIngestJob(id, filename, workspace, (signal) =>
        api.reindexDocument(filename, signal, workspace),
      );
      await refreshAfterIngest(workspace);
    };

    // An earlier original put back (RAW_DOCUMENT_VERSIONS): indexed like an upload of
    // those bytes, and the version it replaces is kept in its turn.
    const handleRestoreVersion = async (
      filename: string,
      version: string,
      workspace = getWorkspace(),
    ) => {
      const id = newId();
      setUploads((prev) => [...prev, { id, filename, status: 'uploading', reindex: true, workspace }]);
      await runIngestJob(id, filename, workspace, (signal) =>
        api.restoreDocumentVersion(filename, version, signal, workspace),
      );
      await refreshAfterIngest(workspace);
    };

    // A document fetched by URL, fetched again. The server asks conditionally: a source
    // that hasn't changed (304) starts no job.
    const handleRefreshFromUrl = async (filename: string, workspace = getWorkspace()) => {
      let result;
      try {
        result = await api.refreshDocument(filename, undefined, workspace);
      } catch (err) {
        if (!deps.current.onAuthFailure(err)) {
          deps.current.pushToast({
            tone: 'bad',
            title: `Couldn't refresh ${filename}`,
            body: apiErrorMessage(err, 'Try again in a moment.'),
          });
        }
        return;
      }
      if (!('job_id' in result)) {
        deps.current.pushToast({ tone: 'good', title: `${filename} is unchanged at its source` });
        return;
      }
      const accepted = result;
      const id = newId();
      setUploads((prev) => [...prev, { id, filename, status: 'uploading', reindex: true, workspace }]);
      await runIngestJob(id, filename, workspace, async () => accepted);
      await refreshAfterIngest(workspace);
    };

    // How often a fetched document is checked again (minutes; 0 never; null the server's).
    const handleSetRefreshInterval = async (
      filename: string,
      minutes: number | null,
      workspace = getWorkspace(),
    ) => {
      try {
        await api.setRefreshInterval(filename, minutes, undefined, workspace);
      } catch (err) {
        if (!deps.current.onAuthFailure(err)) {
          deps.current.pushToast({
            tone: 'bad',
            title: `Couldn't change how often ${filename} is checked`,
            body: apiErrorMessage(err, 'Try again in a moment.'),
          });
        }
        return;
      }
      await refreshDocuments();
    };

    // Several documents chosen at once: started together and followed together, with
    // one listing refresh at the end — handleReindex per document refreshed the whole
    // listing once per document.
    const handleReindexMany = async (filenames: string[], workspace = getWorkspace()) => {
      const items = filenames.map((filename) => ({ id: newId(), filename }));
      setUploads((prev) => [
        ...prev,
        ...items.map(
          ({ id, filename }): UploadItem => ({ id, filename, status: 'uploading', reindex: true, workspace }),
        ),
      ]);
      await Promise.allSettled(
        items.map(({ id, filename }) =>
          runIngestJob(id, filename, workspace, (signal) =>
            api.reindexDocument(filename, signal, workspace),
          ),
        ),
      );
      await refreshAfterIngest(workspace);
    };

    // No arguments: it is an onClick handler as it stands.
    const handleReindexAllStale = async () => {
      const workspace = getWorkspace();
      let response;
      try {
        response = await api.reindexStaleDocuments(undefined, workspace);
      } catch (err) {
        if (!deps.current.onAuthFailure(err)) {
          deps.current.pushToast({
            tone: 'bad',
            title: 'Re-index failed',
            body: apiErrorMessage(err, 'Could not start re-indexing.'),
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
      // Fields an older server doesn't send.
      const failed = response.failed ?? [];
      if (failed.length > 0) {
        deps.current.pushToast({
          tone: 'bad',
          title: `${failed.length} document${failed.length === 1 ? '' : 's'} could not be re-indexed`,
          body: failed.slice(0, 3).join(', ') + (failed.length > 3 ? ', …' : ''),
        });
      }
      const items: UploadItem[] = response.jobs.map((job) => ({
        id: newId(),
        filename: job.filename,
        status: 'uploading',
        reindex: true,
        workspace,
      }));
      setUploads((prev) => [...prev, ...items]);
      await Promise.allSettled(
        response.jobs.map((job, index) =>
          runIngestJob(items[index].id, job.filename, workspace, () => Promise.resolve(job)),
        ),
      );
      await refreshAfterIngest(workspace);
    };

    // Follows jobs the server already started (a backup restore), like uploads.
    // `workspace`: the one the restore was sent to.
    const trackJobs = async (jobs: DocumentJobAcceptedResponse[], workspace = getWorkspace()) => {
      const items: UploadItem[] = jobs.map((job) => ({
        id: newId(),
        filename: job.filename,
        status: 'uploading',
        workspace,
      }));
      setUploads((prev) => [...prev, ...items]);
      await Promise.allSettled(
        jobs.map((job, index) =>
          runIngestJob(items[index].id, job.filename, workspace, () => Promise.resolve(job)),
        ),
      );
      await refreshAfterIngest(workspace);
    };

    const handleUpload = async (files: File[]) => {
      const workspace = getWorkspace();
      const tags = currentUploadTags();
      const newItems: UploadItem[] = files.map((file) => ({
        id: newId(),
        filename: file.name,
        status: 'uploading',
        file,
        workspace,
        tags,
      }));
      setUploads((prev) => [...prev, ...newItems]);
      // A few bodies at a time (see UPLOAD_CONCURRENCY): a 20-file drop used to send 20
      // at once, then all 20 waited and re-sent together when the queue was full. A file
      // used to hold its slot until it was indexed, so the third one wasn't even sent
      // until the first had finished.
      await Promise.allSettled(
        files.map((file, index) => uploadOne(file, newItems[index].id, workspace, tags)),
      );
      // Once for the whole batch, not once per file. Chunk counts live server-side and
      // GET /documents scrolls the entire collection to compute them, so refreshing
      // inside uploadOne meant a 20-file drop triggered 20 full-collection scans.
      await refreshAfterIngest(workspace);
    };

    const handleRetryUpload = async (id: string) => {
      const item = uploadsRef.current.find((candidate) => candidate.id === id);
      if (!item || (!item.file && !item.reindex && !item.sourceUrl)) return;
      setUploads((prev) =>
        prev.map((candidate) =>
          candidate.id === id
            ? { ...candidate, status: 'uploading', error: undefined, progress: undefined }
            : candidate,
        ),
      );
      // Retried where it was first sent, whichever workspace is in use now.
      const { file, filename, sourceUrl, workspace = '', tags } = item;
      await runIngestJob(id, filename, workspace, (signal) =>
        file
          ? sendUpload(file, signal, workspace, tags)
          : sourceUrl
            ? startFromUrl(id, sourceUrl, workspace, tags)(signal)
            : api.reindexDocument(filename, signal, workspace),
      );
      await refreshAfterIngest(workspace);
    };

    const handleDismissUpload = (id: string) => {
      setUploads((prev) => prev.filter((item) => item.id !== id));
    };

    const handleDeleteDocument = async (filename: string, workspace = getWorkspace()) => {
      await api.deleteDocument(filename, undefined, workspace);
      // A deleted document can never remain in a search scope — of that workspace's
      // conversations: another may hold a document of the same name.
      deps.current.removeFromScopes([filename], workspace);
      // Deleted from a workspace no longer in use: the list here is another one's.
      if (!isCurrent(workspace)) return;
      // Any listing already on its way predates the delete.
      listingSequence += 1;
      setIndexedFilenames((prev) => prev.filter((name) => name !== filename));
      setDocuments((prev) => {
        if (!(filename in prev)) return prev;
        const next = { ...prev };
        delete next[filename];
        return next;
      });
    };

    const handleSetTags = async (filename: string, tags: string[], workspace = getWorkspace()) => {
      const result = await api.setDocumentTags(filename, tags, undefined, workspace);
      // Deleted while the request was out: don't bring back a record for it, whose
      // tags would linger in the scope popover. Nor one into another workspace's list.
      if (!isCurrent(workspace) || !indexedRef.current.includes(filename)) return;
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
      resetForWorkspace,
      handleUpload,
      handleRetryUpload,
      handleDismissUpload,
      handleCancelUpload,
      handleUploadFromUrl,
      handleReindex,
      handleRestoreVersion,
      handleRefreshFromUrl,
      handleSetRefreshInterval,
      handleReindexMany,
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
    uploadTags,
    setUploadTags,
    ...actions,
  };
}
