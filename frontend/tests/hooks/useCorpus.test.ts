import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiClientError } from '../../src/api/client';
import type { DocumentJobStatusResponse } from '../../src/api/types';
import { useCorpus } from '../../src/hooks/useCorpus';

const api = vi.hoisted(() => ({
  listDocuments: vi.fn(),
  uploadDocument: vi.fn(),
  pollDocumentJob: vi.fn(),
  cancelDocumentJob: vi.fn(),
  deleteDocument: vi.fn(),
  setDocumentTags: vi.fn(),
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
    expect(api.cancelDocumentJob).toHaveBeenCalledWith('job-1');

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
    expect(result.current.uploads[0].status).toBe('success');
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

    expect(api.cancelDocumentJob).toHaveBeenCalledWith('job-7');
    expect(result.current.uploads).toEqual([]);
  });
});
