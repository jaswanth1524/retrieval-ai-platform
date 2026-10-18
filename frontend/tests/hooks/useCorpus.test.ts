import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiClientError, setWorkspace } from '../../src/api/client';
import type { DocumentJobStatusResponse } from '../../src/api/types';
import { useCorpus } from '../../src/hooks/useCorpus';

const api = vi.hoisted(() => ({
  listDocuments: vi.fn(),
  uploadDocument: vi.fn(),
  uploadFromUrl: vi.fn(),
  pollDocumentJob: vi.fn(),
  cancelDocumentJob: vi.fn(),
  deleteDocument: vi.fn(),
  setDocumentTags: vi.fn(),
  reindexDocument: vi.fn(),
}));
vi.mock('../../src/api/client', async () => {
  const actual = await vi.importActual<typeof import('../../src/api/client')>(
    '../../src/api/client',
  );
  return { ...actual, api };
});

function jobStatus(overrides: Partial<DocumentJobStatusResponse> = {}): DocumentJobStatusResponse {
  return {
    job_id: 'job-1',
    filename: 'a.txt',
    state: 'parsing',
    chunks_total: 0,
    chunks_done: 0,
    error: null,
    result: null,
    ...overrides,
  };
}

function setup() {
  const callbacks = {
    onAuthFailure: vi.fn(() => false),
    onAuthRestored: vi.fn(),
    pushToast: vi.fn(),
    pruneScopes: vi.fn(),
    removeFromScopes: vi.fn(),
  };
  const hook = renderHook(() => useCorpus(callbacks));
  return { ...hook, callbacks };
}

const file = new File(['x'], 'a.txt', { type: 'text/plain' });

describe('useCorpus', () => {
  beforeEach(() => {
    for (const mock of Object.values(api)) mock.mockReset();
    api.listDocuments.mockResolvedValue({ filenames: [] });
  });

  it('indexes a URL like an upload, names the card after the server, and retries the URL', async () => {
    const url = 'https://example.com/docs/policy';
    api.uploadFromUrl
      .mockRejectedValueOnce(new ApiClientError('The URL answered 503; nothing was fetched.', 400))
      .mockResolvedValueOnce({ job_id: 'job-1', filename: 'policy.html', state: 'queued' });
    api.pollDocumentJob.mockResolvedValue(
      jobStatus({ state: 'done', filename: 'policy.html', result: { filename: 'policy.html' } as never }),
    );
    const { result } = setup();

    await act(() => result.current.handleUploadFromUrl(url));
    const failed = result.current.uploads[0];
    expect(failed).toMatchObject({ filename: url, status: 'error', sourceUrl: url });
    expect(failed.error).toContain('answered 503');

    api.listDocuments.mockResolvedValue({ filenames: ['policy.html'] });
    await act(() => result.current.handleRetryUpload(failed.id));
    expect(api.uploadFromUrl).toHaveBeenLastCalledWith(url, expect.anything(), '');
    // Done: its card gives way to the document's own once the listing has it.
    expect(result.current.uploads).toEqual([]);
    expect(result.current.indexedFilenames).toEqual(['policy.html']);
  });

  it('keeps one metadata record per document, in the listing order', async () => {
    api.listDocuments.mockResolvedValue({
      filenames: ['2024.pdf', 'b.md'],
      chunk_counts: { '2024.pdf': 3, 'b.md': 2 },
      page_counts: { '2024.pdf': 4 },
      stale_filenames: ['b.md'],
      reindexable_filenames: ['b.md'],
      tags: { 'b.md': ['hr', 'Legal'] },
    });
    const { result } = setup();

    await act(() => result.current.refreshDocuments());

    expect(result.current.indexedFilenames).toEqual(['2024.pdf', 'b.md']);
    expect(result.current.documents['2024.pdf']).toEqual({
      chunks: 3,
      pages: 4,
      bytes: undefined,
      uploadedAt: undefined,
      stale: false,
      reindexable: false,
      tags: [],
    });
    expect(result.current.documents['b.md']).toMatchObject({ stale: true, reindexable: true });
    expect(result.current.chunkTotal).toBe(5);
    expect(result.current.availableTags).toEqual(['hr', 'Legal']);

    api.deleteDocument.mockResolvedValue({ filename: 'b.md', points_deleted: 2 });
    await act(() => result.current.handleDeleteDocument('b.md'));
    expect(result.current.documents).not.toHaveProperty('b.md');
    expect(result.current.availableTags).toEqual([]);
  });

  it('ignores a listing that was overtaken by a delete', async () => {
    api.listDocuments.mockResolvedValueOnce({ filenames: ['a.md', 'b.md'], chunk_counts: {} });
    const { result } = setup();
    await act(() => result.current.refreshDocuments());

    // A slow listing, requested before the delete, answers after it.
    let answerSlowListing!: (value: unknown) => void;
    api.listDocuments.mockReturnValueOnce(new Promise((resolve) => (answerSlowListing = resolve)));
    let slow!: Promise<void>;
    act(() => {
      slow = result.current.refreshDocuments();
    });
    api.deleteDocument.mockResolvedValue({ filename: 'b.md', points_deleted: 1 });
    await act(() => result.current.handleDeleteDocument('b.md'));
    answerSlowListing({ filenames: ['a.md', 'b.md'], chunk_counts: {}, tags: { 'b.md': ['x'] } });
    await act(() => slow);

    expect(result.current.indexedFilenames).toEqual(['a.md']);

    // Nor does a tag change that lands after the delete bring a record back.
    api.setDocumentTags.mockResolvedValue({ filename: 'b.md', tags: ['late'] });
    await act(() => result.current.handleSetTags('b.md', ['late']));
    expect(result.current.documents).not.toHaveProperty('b.md');
    expect(result.current.availableTags).toEqual([]);
  });

  it('cancels an upload still waiting for room in the queue without asking the server', async () => {
    api.uploadDocument.mockRejectedValue(
      new ApiClientError('Too many documents are already waiting.', 503, 15),
    );
    const { result } = setup();

    let upload!: Promise<void>;
    act(() => {
      upload = result.current.handleUpload([file]);
    });
    await waitFor(() => expect(result.current.uploads[0]?.progress?.state).toBe('waiting'));

    act(() => result.current.handleCancelUpload(result.current.uploads[0].id));
    await act(() => upload);

    expect(result.current.uploads).toEqual([]);
    // Woken early and stopped: not retried, and there was no job to cancel.
    expect(api.uploadDocument).toHaveBeenCalledTimes(1);
    expect(api.cancelDocumentJob).not.toHaveBeenCalled();
  });

  it('stops promptly when Cancel lands while the upload is sent and the queue is full', async () => {
    let reject!: (err: unknown) => void;
    api.uploadDocument.mockReturnValue(new Promise((_, fail) => (reject = fail)));
    const { result } = setup();

    let upload!: Promise<void>;
    act(() => {
      upload = result.current.handleUpload([file]);
    });
    act(() => result.current.handleCancelUpload(result.current.uploads[0].id));
    // Nothing is waiting to be woken yet: the 503 itself must stop it, not 15 s later.
    reject(new ApiClientError('Too many documents are already waiting.', 503, 15));
    await act(() => upload);

    expect(result.current.uploads).toEqual([]);
    expect(api.uploadDocument).toHaveBeenCalledTimes(1);
    expect(api.cancelDocumentJob).not.toHaveBeenCalled();
  });

  it('aborts the request still sending the file when Cancel is pressed', async () => {
    // Cancel only took effect after the whole body was sent: minutes for a big file on a
    // slow link, with the card saying "cancelling" all along.
    let sentSignal: AbortSignal | undefined;
    api.uploadDocument.mockImplementation(
      (_file: File, signal?: AbortSignal) =>
        new Promise((_, reject) => {
          sentSignal = signal;
          signal?.addEventListener('abort', () =>
            reject(new ApiClientError('Request timed out or was cancelled.')),
          );
        }),
    );
    const { result } = setup();

    let upload!: Promise<void>;
    act(() => {
      upload = result.current.handleUpload([file]);
    });
    await waitFor(() => expect(sentSignal).toBeDefined());
    act(() => result.current.handleCancelUpload(result.current.uploads[0].id));
    await act(() => upload);

    expect(sentSignal?.aborted).toBe(true);
    expect(result.current.uploads).toEqual([]); // removed, not shown as a failure
    expect(api.cancelDocumentJob).not.toHaveBeenCalled();
  });

  it('asks the server to stop an accepted job and drops the card once it reports cancelled', async () => {
    api.uploadDocument.mockResolvedValue({ job_id: 'job-1', filename: 'a.txt', state: 'queued' });
    let finishPoll!: (status: DocumentJobStatusResponse) => void;
    api.pollDocumentJob.mockReturnValue(new Promise((resolve) => (finishPoll = resolve)));
    api.cancelDocumentJob.mockResolvedValue(jobStatus({ state: 'queued' }));
    const { result } = setup();

    let upload!: Promise<void>;
    act(() => {
      upload = result.current.handleUpload([file]);
    });
    await waitFor(() => expect(api.pollDocumentJob).toHaveBeenCalled());

    act(() => result.current.handleCancelUpload(result.current.uploads[0].id));
    expect(result.current.uploads[0].cancelling).toBe(true);
    expect(api.cancelDocumentJob).toHaveBeenCalledWith('job-1', undefined, '');

    finishPoll(jobStatus({ state: 'failed', cancelled: true, error: 'Cancelled.' }));
    await act(() => upload);

    expect(result.current.uploads).toEqual([]);
  });

  it('keeps following a job the server refuses to cancel, and says why', async () => {
    api.uploadDocument.mockResolvedValue({ job_id: 'job-1', filename: 'a.txt', state: 'queued' });
    let finishPoll!: (status: DocumentJobStatusResponse) => void;
    api.pollDocumentJob.mockReturnValue(new Promise((resolve) => (finishPoll = resolve)));
    api.cancelDocumentJob.mockRejectedValue(
      new ApiClientError('This job is already writing to the index.', 409),
    );
    const { result, callbacks } = setup();

    let upload!: Promise<void>;
    act(() => {
      upload = result.current.handleUpload([file]);
    });
    await waitFor(() => expect(api.pollDocumentJob).toHaveBeenCalled());

    act(() => result.current.handleCancelUpload(result.current.uploads[0].id));
    await waitFor(() => expect(result.current.uploads[0].cancelling).toBe(false));
    expect(callbacks.pushToast).toHaveBeenCalledWith(
      expect.objectContaining({
        tone: 'warn',
        title: "Couldn't cancel a.txt",
        body: 'This job is already writing to the index.',
      }),
    );

    finishPoll(
      jobStatus({
        state: 'done',
        result: {
          filename: 'a.txt',
          collection_name: 'documents',
          sections_parsed: 1,
          chunks_ingested: 1,
        } as DocumentJobStatusResponse['result'],
      }),
    );
    await act(() => upload);
    // Indexed and listed: the finished card is dropped rather than kept all session.
    expect(result.current.uploads).toEqual([]);
  });

  it('cancels a job the server accepts after Cancel was pressed mid-upload', async () => {
    let accept!: (value: { job_id: string; filename: string; state: 'queued' }) => void;
    api.uploadDocument.mockReturnValue(new Promise((resolve) => (accept = resolve)));
    api.pollDocumentJob.mockResolvedValue(jobStatus({ state: 'failed', cancelled: true }));
    api.cancelDocumentJob.mockResolvedValue(jobStatus({ state: 'queued' }));
    const { result } = setup();

    let upload!: Promise<void>;
    act(() => {
      upload = result.current.handleUpload([file]);
    });
    act(() => result.current.handleCancelUpload(result.current.uploads[0].id));
    accept({ job_id: 'job-7', filename: 'a.txt', state: 'queued' });
    await act(() => upload);

    expect(api.cancelDocumentJob).toHaveBeenCalledWith('job-7', undefined, '');
    expect(result.current.uploads).toEqual([]);
  });

  it('sends the next file while earlier ones are still being indexed', async () => {
    // Two send slots. Each accepted job then polls forever until released, which used
    // to keep its slot: the third file was not sent until the first was indexed.
    const finish: Array<() => void> = [];
    api.uploadDocument.mockImplementation(async (sent: File) => ({
      job_id: `job-${sent.name}`,
      filename: sent.name,
      state: 'queued',
    }));
    api.pollDocumentJob.mockImplementation(
      (jobId: string) =>
        new Promise((resolve) =>
          finish.push(() => resolve(jobStatus({ job_id: jobId, state: 'done' }))),
        ),
    );
    const { result } = setup();
    const files = ['a.txt', 'b.txt', 'c.txt'].map((name) => new File(['x'], name));

    let upload: Promise<void> | undefined;
    act(() => {
      upload = result.current.handleUpload(files);
    });

    await waitFor(() => expect(api.uploadDocument).toHaveBeenCalledTimes(3));
    expect(api.pollDocumentJob).toHaveBeenCalledTimes(3);
    await act(async () => {
      for (const done of finish) done();
      await upload;
    });
  });

  it('sends retried files through the same two send slots', async () => {
    api.uploadDocument.mockRejectedValue(new ApiClientError('Server busy.', 500));
    const { result } = setup();
    const files = ['a.txt', 'b.txt', 'c.txt'].map((name) => new File(['x'], name));
    await act(() => result.current.handleUpload(files));
    expect(result.current.uploads.map((item) => item.status)).toEqual(['error', 'error', 'error']);

    let sending = 0;
    let most = 0;
    const release: Array<() => void> = [];
    api.uploadDocument.mockReset();
    api.uploadDocument.mockImplementation(
      (sent: File) =>
        new Promise((resolve) => {
          sending += 1;
          most = Math.max(most, sending);
          release.push(() => {
            sending -= 1;
            resolve({ job_id: `job-${sent.name}`, filename: sent.name, state: 'queued' });
          });
        }),
    );
    api.pollDocumentJob.mockImplementation(async (jobId: string) =>
      jobStatus({ job_id: jobId, state: 'done' }),
    );

    let retries: Promise<unknown> | undefined;
    act(() => {
      retries = Promise.all(
        result.current.uploads.map((item) => result.current.handleRetryUpload(item.id)),
      );
    });
    await waitFor(() => expect(api.uploadDocument).toHaveBeenCalledTimes(2));
    await act(async () => {
      while (release.length > 0) {
        release.shift()?.();
        await Promise.resolve();
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
      await retries;
    });

    expect(api.uploadDocument).toHaveBeenCalledTimes(3);
    expect(most).toBe(2);
  });
});

describe('useCorpus across workspaces', () => {
  beforeEach(() => {
    for (const mock of Object.values(api)) mock.mockReset();
    api.listDocuments.mockResolvedValue({ filenames: [] });
    setWorkspace('legal');
  });
  afterEach(() => setWorkspace(''));

  const done = (filename: string) =>
    jobStatus({ state: 'done', filename, result: { filename } as DocumentJobStatusResponse['result'] });

  it('sends an upload waiting for room in the queue to the workspace it was started in', async () => {
    api.uploadDocument
      .mockRejectedValueOnce(new ApiClientError('The indexing queue is full.', 503, 1))
      .mockResolvedValueOnce({ job_id: 'job-1', filename: 'a.txt', state: 'queued' });
    api.pollDocumentJob.mockResolvedValue(done('a.txt'));
    const { result } = setup();

    let upload!: Promise<void>;
    act(() => {
      upload = result.current.handleUpload([file]);
    });
    await waitFor(() => expect(result.current.uploads[0]?.progress?.state).toBe('waiting'));
    expect(result.current.uploads[0].workspace).toBe('legal');
    // Switched to the default while it waits.
    setWorkspace('');
    act(() => result.current.resetForWorkspace());
    await act(() => upload);

    expect(api.uploadDocument.mock.calls.map((call) => call[2])).toEqual(['legal', 'legal']);
    expect(api.pollDocumentJob).toHaveBeenCalledWith('job-1', expect.any(Function), undefined, 'legal');
    // Indexed in legal: the default workspace's list doesn't gain it, nor is it relisted.
    expect(result.current.indexedFilenames).toEqual([]);
    expect(api.listDocuments).not.toHaveBeenCalled();
    expect(result.current.uploads).toEqual([]);
  });

  it('retries a failed upload in its own workspace, whichever is in use', async () => {
    api.uploadDocument
      .mockRejectedValueOnce(new ApiClientError('Ingest backend down.', 500))
      .mockResolvedValueOnce({ job_id: 'job-2', filename: 'a.txt', state: 'queued' });
    api.pollDocumentJob.mockResolvedValue(done('a.txt'));
    const { result } = setup();
    await act(() => result.current.handleUpload([file]));
    expect(result.current.uploads[0].status).toBe('error');

    setWorkspace('');
    await act(() => result.current.handleRetryUpload(result.current.uploads[0].id));

    expect(api.uploadDocument).toHaveBeenLastCalledWith(file, expect.anything(), 'legal');
  });

  it('leaves the list of the workspace now in use alone when a delete or tag change ends', async () => {
    api.listDocuments.mockResolvedValue({ filenames: ['shared.md'], tags: { 'shared.md': ['a'] } });
    let finishDelete!: () => void;
    api.deleteDocument.mockReturnValue(new Promise<void>((resolve) => (finishDelete = resolve)));
    let finishTags!: (value: { filename: string; tags: string[] }) => void;
    api.setDocumentTags.mockReturnValue(new Promise((resolve) => (finishTags = resolve)));
    const { result, callbacks } = setup();
    await act(() => result.current.refreshDocuments());

    let deleting!: Promise<void>;
    let tagging!: Promise<void>;
    act(() => {
      deleting = result.current.handleDeleteDocument('shared.md');
      tagging = result.current.handleSetTags('shared.md', ['b']);
    });
    // The default workspace holds a document of the same name.
    setWorkspace('');
    act(() => result.current.resetForWorkspace());
    await act(() => result.current.refreshDocuments());
    finishDelete();
    finishTags({ filename: 'shared.md', tags: ['b'] });
    await act(() => Promise.all([deleting, tagging]));

    expect(api.deleteDocument).toHaveBeenCalledWith('shared.md', undefined, 'legal');
    expect(api.setDocumentTags).toHaveBeenCalledWith('shared.md', ['b'], undefined, 'legal');
    expect(result.current.indexedFilenames).toEqual(['shared.md']);
    expect(result.current.documents['shared.md'].tags).toEqual(['a']);
    // Pruned from legal's conversations, not the default workspace's.
    expect(callbacks.removeFromScopes).toHaveBeenCalledWith(['shared.md'], 'legal');
  });

  it("drops a listing of the previous workspace that lands after a switch", async () => {
    let answer!: (value: { filenames: string[] }) => void;
    api.listDocuments.mockReturnValueOnce(new Promise((resolve) => (answer = resolve)));
    const { result } = setup();

    let listing!: Promise<void>;
    act(() => {
      listing = result.current.refreshDocuments();
    });
    setWorkspace('');
    answer({ filenames: ['contract.txt'] });
    await act(() => listing);

    expect(result.current.indexedFilenames).toEqual([]);
  });

  it('forgets the previous workspace’s corpus on a switch, keeping unfinished cards', async () => {
    api.listDocuments.mockResolvedValue({ filenames: ['contract.txt'], chunk_counts: { 'contract.txt': 2 } });
    api.uploadDocument.mockRejectedValue(new ApiClientError('Ingest backend down.', 500));
    const { result } = setup();
    await act(() => result.current.refreshDocuments());
    await act(() => result.current.handleUpload([file]));

    act(() => result.current.resetForWorkspace());

    expect(result.current.indexedFilenames).toEqual([]);
    expect(result.current.documents).toEqual({});
    expect(result.current.uploads).toEqual([
      expect.objectContaining({ status: 'error', workspace: 'legal' }),
    ]);
  });
});
