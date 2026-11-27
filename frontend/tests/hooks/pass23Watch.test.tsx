import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from '../../src/api/client';
import { ApiClientError } from '../../src/api/errors';
import type { RecheckResponse } from '../../src/api/types';
import type { ChatTurn } from '../../src/components/ChatMessage';
import type { DocumentMeta } from '../../src/components/corpus/types';
import { SAVED_WATCH_MAX, useSavedAnswerWatch } from '../../src/hooks/useSavedAnswerWatch';

afterEach(() => vi.restoreAllMocks());

function meta(uploadedAt: number, extra: Partial<DocumentMeta> = {}): DocumentMeta {
  return { chunks: 1, uploadedAt, stale: false, reindexable: true, tags: [], ...extra } as DocumentMeta;
}

function saved(filename: string, id = filename) {
  const turn: ChatTurn = {
    id,
    role: 'assistant',
    content: 'Staff get 28 days [1].',
    sources: [{ source_number: 1, filename, page: 1, section: 'Leave', chunk_id: 'c1', text: 'x' }],
    timestamp: 1,
    timings: null,
    traceId: null,
    saved: true,
  };
  return { conversationId: 'c', conversationTitle: 'Leave', workspace: '', turn, question: 'Leave?' };
}

const BROKEN: RecheckResponse = {
  sources: [],
  sentences: [{ sentence: 'x', status: 'unsupported' }],
  still_holds: false,
};

const tick = () => new Promise((resolve) => setTimeout(resolve, 20));

function watch(answers: ReturnType<typeof saved>[], documents: Record<string, DocumentMeta>) {
  const told = vi.fn();
  const hook = renderHook(
    (props: { documents: Record<string, DocumentMeta>; workspace?: string }) =>
      useSavedAnswerWatch({
        answers,
        documents: props.documents,
        ready: true,
        workspace: props.workspace ?? '',
        onNoLongerHold: told,
      }),
    { initialProps: { documents } as { documents: Record<string, DocumentMeta>; workspace?: string } },
  );
  return { told, ...hook };
}

describe('pass 23: the saved-answer watch', () => {
  it('doesn’t wait for a document that stays incomplete', async () => {
    const recheck = vi.spyOn(api, 'recheckAnswer').mockResolvedValue(BROKEN);
    const { told, rerender } = watch([saved('a.txt'), saved('b.txt')], {
      'a.txt': meta(1),
      'b.txt': meta(1),
    });
    rerender({ documents: { 'a.txt': meta(2, { incomplete: true }), 'b.txt': meta(1) } });
    rerender({ documents: { 'a.txt': meta(2, { incomplete: true }) } });
    await waitFor(() => expect(told).toHaveBeenCalledWith(1, ['b.txt'], 0));
    expect(recheck).toHaveBeenCalledTimes(1);
  });

  it('goes on through a listing that changes nothing it watches', async () => {
    const replies: Array<(value: RecheckResponse) => void> = [];
    const recheck = vi
      .spyOn(api, 'recheckAnswer')
      .mockImplementation(() => new Promise((resolve) => replies.push(resolve)));
    const { told, rerender } = watch([saved('a.txt')], { 'a.txt': meta(1) });
    rerender({ documents: { 'a.txt': meta(2) } });
    rerender({ documents: { 'a.txt': meta(2, { tags: ['x'] }) } });
    expect(recheck).toHaveBeenCalledTimes(1);
    // The same document again while the run goes: it starts over for the newer one.
    rerender({ documents: { 'a.txt': meta(3) } });
    expect(recheck).toHaveBeenCalledTimes(2);
    await act(async () => replies[1](BROKEN));
    expect(told).toHaveBeenCalledTimes(1);
  });

  it('says how many it didn’t check', async () => {
    vi.spyOn(api, 'recheckAnswer').mockResolvedValue(BROKEN);
    const answers = Array.from({ length: SAVED_WATCH_MAX + 3 }, (_, n) => saved('a.txt', `t${n}`));
    const { told, rerender } = watch(answers, { 'a.txt': meta(1) });
    rerender({ documents: { 'a.txt': meta(2) } });
    await waitFor(() => expect(told).toHaveBeenCalledWith(SAVED_WATCH_MAX, ['a.txt'], 3));
  });

  it('says nothing of a part, and tries again with the next change', async () => {
    const recheck = vi
      .spyOn(api, 'recheckAnswer')
      .mockResolvedValueOnce(BROKEN)
      .mockRejectedValueOnce(new ApiClientError('busy', 429))
      .mockResolvedValue(BROKEN);
    const { told, rerender } = watch([saved('a.txt', 't1'), saved('a.txt', 't2')], {
      'a.txt': meta(1),
    });
    rerender({ documents: { 'a.txt': meta(2) } });
    await tick();
    expect(told).not.toHaveBeenCalled();
    rerender({ documents: { 'a.txt': meta(2), 'z.txt': meta(1) } });
    await waitFor(() => expect(told).toHaveBeenCalledWith(2, ['a.txt'], 0));
    expect(recheck).toHaveBeenCalledTimes(4);
  });

  it('stops asking a server without the route', async () => {
    const recheck = vi.spyOn(api, 'recheckAnswer').mockRejectedValue(new ApiClientError('no', 404));
    const { rerender } = watch([saved('a.txt')], { 'a.txt': meta(1) });
    rerender({ documents: { 'a.txt': meta(2) } });
    await tick();
    rerender({ documents: { 'a.txt': meta(3) } });
    await tick();
    expect(recheck).toHaveBeenCalledTimes(1);
  });

  it('drops a run when another workspace is opened', async () => {
    let signal: AbortSignal | undefined;
    vi.spyOn(api, 'recheckAnswer').mockImplementation((_, given) => {
      signal = given;
      return new Promise(() => {});
    });
    const { rerender } = watch([saved('a.txt')], { 'a.txt': meta(1) });
    rerender({ documents: { 'a.txt': meta(2) } });
    rerender({ documents: {}, workspace: 'other' });
    expect(signal?.aborted).toBe(true);
  });
});
