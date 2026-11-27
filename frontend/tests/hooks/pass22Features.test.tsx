import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from '../../src/api/client';
import type { ChatTurn } from '../../src/components/ChatMessage';
import type { DocumentMeta } from '../../src/components/corpus/types';
import { useOtherTabAnswering } from '../../src/hooks/useOtherTabAnswering';
import { useSavedAnswerWatch } from '../../src/hooks/useSavedAnswerWatch';
import { nextPlaceholder } from '../../src/utils/questionTemplates';

afterEach(() => vi.restoreAllMocks());

function meta(uploadedAt: number): DocumentMeta {
  return { chunks: 1, uploadedAt, stale: false, reindexable: true, tags: [] } as DocumentMeta;
}

function saved(filename: string) {
  const turn: ChatTurn = {
    id: `t-${filename}`,
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

describe('pass 22: saved answers watched', () => {
  it('are rechecked when a document they cite changes, and the reader told', async () => {
    const recheck = vi
      .spyOn(api, 'recheckAnswer')
      .mockResolvedValue({
        sources: [{ source_number: 1, filename: 'leave.txt', chunk_id: 'c1', state: 'changed' }],
        sentences: [{ sentence: 'Staff get 28 days [1].', status: 'numbers_differ' }],
        still_holds: false,
      });
    const told = vi.fn();
    const answers = [saved('leave.txt'), saved('other.txt')];
    const { rerender } = renderHook(
      ({ documents }) =>
        useSavedAnswerWatch({ answers, documents, ready: true, workspace: '', onNoLongerHold: told }),
      { initialProps: { documents: { 'leave.txt': meta(1), 'other.txt': meta(1) } } },
    );
    expect(recheck).not.toHaveBeenCalled(); // the first listing is no change
    rerender({ documents: { 'leave.txt': meta(2), 'other.txt': meta(1) } });
    await waitFor(() => expect(told).toHaveBeenCalledWith(1, ['leave.txt'], 0));
    expect(recheck).toHaveBeenCalledTimes(1);
  });

  it('says nothing when they still hold', async () => {
    vi.spyOn(api, 'recheckAnswer').mockResolvedValue({ sources: [], sentences: [], still_holds: true });
    const told = vi.fn();
    const answers = [saved('leave.txt')];
    const { rerender } = renderHook(
      ({ documents }) =>
        useSavedAnswerWatch({ answers, documents, ready: true, workspace: '', onNoLongerHold: told }),
      { initialProps: { documents: { 'leave.txt': meta(1) } as Record<string, DocumentMeta> } },
    );
    rerender({ documents: {} });
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(told).not.toHaveBeenCalled();
  });
});

describe('pass 22: saved answers watched, after the review', () => {
  it('don\'t count a long answer whose checked sentences all hold', async () => {
    vi.spyOn(api, 'recheckAnswer').mockResolvedValue({
      sources: [{ source_number: 1, filename: 'leave.txt', chunk_id: 'c1', state: 'changed' }],
      sentences: [{ sentence: 'Staff get 28 days [1].', status: 'supported' }],
      still_holds: false,
      more: true,
    });
    const told = vi.fn();
    const answers = [saved('leave.txt')];
    const { rerender } = renderHook(
      ({ documents }) =>
        useSavedAnswerWatch({ answers, documents, ready: true, workspace: '', onNoLongerHold: told }),
      { initialProps: { documents: { 'leave.txt': meta(1) } } },
    );
    rerender({ documents: { 'leave.txt': meta(2) } });
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(told).not.toHaveBeenCalled();
  });

  it('wait for a document still being written', async () => {
    const recheck = vi.spyOn(api, 'recheckAnswer').mockResolvedValue({
      sources: [],
      sentences: [{ sentence: 'x', status: 'unsupported' }],
      still_holds: false,
    });
    const told = vi.fn();
    const answers = [saved('leave.txt')];
    const { rerender } = renderHook(
      ({ documents }) =>
        useSavedAnswerWatch({ answers, documents, ready: true, workspace: '', onNoLongerHold: told }),
      { initialProps: { documents: { 'leave.txt': meta(1) } } },
    );
    rerender({ documents: { 'leave.txt': { ...meta(2), incomplete: true } } });
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(recheck).not.toHaveBeenCalled();
    rerender({ documents: { 'leave.txt': meta(2) } });
    await waitFor(() => expect(told).toHaveBeenCalledWith(1, ['leave.txt'], 0));
  });
});

describe('pass 22: placeholders', () => {
  it('Tab goes to the next one, and past the last leaves the box', () => {
    const text = 'Compare {a} with {b}.';
    expect(nextPlaceholder(text, 0)).toEqual([8, 11]);
    expect(nextPlaceholder(text, 11)).toEqual([17, 20]);
    expect(nextPlaceholder(text, 20)).toBeNull();
    expect(nextPlaceholder('none here', 0)).toBeNull();
  });
});

describe('pass 22: another tab answering', () => {
  it('is seen in the conversation it answers in, and no longer once it stops', async () => {
    const { result } = renderHook(() => useOtherTabAnswering('c1', null));
    const channel = new BroadcastChannel('docrag-chat');
    act(() => channel.postMessage({ type: 'answering', from: 'tab-b', conversationId: 'c1' }));
    await waitFor(() => expect(result.current).toBe(true));
    act(() => channel.postMessage({ type: 'answering', from: 'tab-b', conversationId: null }));
    await waitFor(() => expect(result.current).toBe(false));
    channel.close();
  });
});
