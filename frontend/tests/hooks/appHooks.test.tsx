import { act, render, renderHook, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiClientError, api } from '../../src/api/client';
import type { PublicConfigResponse } from '../../src/api/types';
import ShortcutsDialog from '../../src/components/ShortcutsDialog';
import { useAppBoot } from '../../src/hooks/useAppBoot';
import { answeredTurn, countDown, withBusyRetry } from '../../src/hooks/useChat';
import { useConnectionMonitor } from '../../src/hooks/useConnectionMonitor';
import { useInspectorVisibility } from '../../src/hooks/useInspectorVisibility';
import type { ChatTurn } from '../../src/components/ChatMessage';
import { safeFilename } from '../../src/utils/exportChat';

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe('useAppBoot', () => {
  it('boots with the config, the access level and the first listing', async () => {
    const config = { llm_model: 'qwen2.5' } as PublicConfigResponse;
    vi.spyOn(api, 'config').mockResolvedValue(config);
    vi.spyOn(api, 'access').mockResolvedValue({ access: 'read' });
    const refreshDocuments = vi.fn(async () => {});
    const onConfig = vi.fn();

    const { result } = renderHook(() => useAppBoot({ refreshDocuments, onConfig }));

    await waitFor(() => expect(result.current.apiStatus).toBe('ok'));
    await waitFor(() => expect(result.current.corpusChecked).toBe(true));
    expect(result.current.config).toBe(config);
    expect(onConfig).toHaveBeenCalledWith(config);
    expect(refreshDocuments).toHaveBeenCalledTimes(1);
    expect(result.current.access).toBe('read');
    expect(result.current.listingFailed).toBe(false);
  });

  it('a listing that fails leaves the corpus unknown, and a 401 shows the key banner', async () => {
    vi.spyOn(api, 'config').mockResolvedValue({} as PublicConfigResponse);
    vi.spyOn(api, 'access').mockResolvedValue({ access: 'full' });
    const refreshDocuments = vi.fn(async () => {
      throw new ApiClientError('Server error', 500);
    });

    const { result } = renderHook(() => useAppBoot({ refreshDocuments, onConfig: () => {} }));

    await waitFor(() => expect(result.current.listingFailed).toBe(true));
    expect(result.current.corpusChecked).toBe(true);
    act(() => {
      expect(result.current.noteAuthFailure(new ApiClientError('Key', 401))).toBe(true);
    });
    expect(result.current.apiStatus).toBe('unauthorized');
    act(() => result.current.noteAuthRestored());
    expect(result.current.apiStatus).toBe('ok');
  });
});

describe('useConnectionMonitor', () => {
  it('notices the server going away and coming back', async () => {
    vi.useFakeTimers();
    const health = vi.spyOn(api, 'health').mockRejectedValue(new ApiClientError('down'));
    const onVersion = vi.fn();
    const { result } = renderHook(() => useConnectionMonitor(true, onVersion));
    expect(result.current).toBe(false);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });
    expect(result.current).toBe(true);

    health.mockResolvedValue({ status: 'ok', version: '0.2.0' });
    await act(async () => {
      // Checked again sooner once it has failed.
      await vi.advanceTimersByTimeAsync(5_000);
    });
    expect(result.current).toBe(false);
    expect(onVersion).toHaveBeenCalledWith('0.2.0');

    act(() => {
      window.dispatchEvent(new Event('offline'));
    });
    expect(result.current).toBe(true);
  });

  it('polls nothing before boot', async () => {
    vi.useFakeTimers();
    const health = vi.spyOn(api, 'health');
    renderHook(() => useConnectionMonitor(false));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000);
    });
    expect(health).not.toHaveBeenCalled();
  });
});

describe('useInspectorVisibility', () => {
  it('follows the width until the reader takes a position', () => {
    const onReopen = vi.fn();
    const { result, rerender } = renderHook(
      ({ narrow, roomy }) => useInspectorVisibility(narrow, roomy, onReopen),
      { initialProps: { narrow: false, roomy: true } },
    );
    expect(result.current.inspectorOpen).toBe(true);
    rerender({ narrow: true, roomy: false });
    expect(result.current.inspectorOpen).toBe(false);

    // Opened by hand on a narrow window: it stays open, and reopening said so.
    act(() => result.current.toggleInspector());
    expect(result.current.inspectorOpen).toBe(true);
    expect(onReopen).toHaveBeenCalledTimes(1);
    rerender({ narrow: false, roomy: true });
    act(() => result.current.closeInspector());
    rerender({ narrow: false, roomy: true });
    expect(result.current.inspectorOpen).toBe(false);
  });

  it('an inspector opened wide still closes when the window narrows', () => {
    const { result, rerender } = renderHook(
      ({ narrow, roomy }) => useInspectorVisibility(narrow, roomy),
      { initialProps: { narrow: false, roomy: false } },
    );
    act(() => result.current.showInspector());
    expect(result.current.inspectorOpen).toBe(true);
    rerender({ narrow: true, roomy: false });
    expect(result.current.inspectorOpen).toBe(false);
  });
});

describe('countDown and withBusyRetry', () => {
  it('counts down each second and ends early when stopped', async () => {
    vi.useFakeTimers();
    const ticks: number[] = [];
    const done = countDown(3, new AbortController().signal, (left) => ticks.push(left));
    await vi.advanceTimersByTimeAsync(3000);
    await done;
    expect(ticks).toEqual([3, 2, 1]);

    const stop = new AbortController();
    const stopped = countDown(30, stop.signal, () => {});
    stop.abort();
    await expect(stopped).resolves.toBeUndefined();
  });

  it('asks again after a busy refusal, but not after any other error', async () => {
    vi.useFakeTimers();
    const send = vi
      .fn<() => Promise<void>>()
      .mockRejectedValueOnce(new ApiClientError('Busy', 429, 1))
      .mockResolvedValueOnce();
    const resumed = vi.fn();
    const asked = withBusyRetry(send, new AbortController().signal, () => {}, resumed);
    await vi.advanceTimersByTimeAsync(1000);
    await asked;
    expect(send).toHaveBeenCalledTimes(2);
    expect(resumed).toHaveBeenCalledTimes(1);

    const failing = vi.fn(async () => {
      throw new ApiClientError('boom', 500);
    });
    await expect(
      withBusyRetry(failing, new AbortController().signal, () => {}, () => {}),
    ).rejects.toThrow('boom');
    expect(failing).toHaveBeenCalledTimes(1);
  });
});

describe('answeredTurn and safeFilename', () => {
  it('maps the done frame onto the answer turn', () => {
    const turn: ChatTurn = {
      id: 'a',
      role: 'assistant',
      content: '',
      sources: [],
      timestamp: 1,
      timings: null,
      traceId: null,
    };
    const done = answeredTurn(
      turn,
      'Not in the documents.',
      [],
      { embed_ms: 0, search_ms: 0, rerank_ms: 0, generate_ms: 0, total_ms: 0, condense_ms: 0 },
      't1',
      true,
      { answerStatus: 'no_context', weakMatches: true, usage: { prompt_tokens: 3, completion_tokens: 2 } },
    );
    expect(done).toMatchObject({
      content: 'Not in the documents.',
      traceId: 't1',
      cached: true,
      notFound: true,
      weakMatch: true,
      usage: { prompt: 3, completion: 2 },
    });
    expect(done.uncited).toBeUndefined();
  });

  it('turns a title into a file name', () => {
    expect(safeFilename('docrag-Leave policy: 2024?.md')).toBe('docrag-Leave-policy-2024-.md');
  });
});

describe('ShortcutsDialog', () => {
  it('lists every bound shortcut and the plain keys, and closes', async () => {
    const onClose = vi.fn();
    render(
      <ShortcutsDialog
        commands={[
          { id: 'new', label: 'New chat', glyph: '+', shortcut: '⌘⇧O', run: () => {} },
          { id: 'help', label: 'Keyboard shortcuts', glyph: '?', shortcut: '?', run: () => {} },
          { id: 'plain', label: 'No shortcut', glyph: '·', run: () => {} },
        ]}
        onClose={onClose}
      />,
    );
    const dialog = screen.getByRole('dialog', { name: 'Keyboard shortcuts' });
    expect(dialog).toHaveTextContent('Command palette');
    expect(dialog).toHaveTextContent('New chat');
    expect(dialog).not.toHaveTextContent('No shortcut');
    // "?" itself is among the plain keys, not repeated as a command.
    expect(screen.queryAllByText('Keyboard shortcuts', { selector: 'td' })).toHaveLength(0);

    await userEvent.keyboard('{Escape}');
    expect(onClose).toHaveBeenCalledTimes(1);
    await userEvent.click(screen.getByRole('button', { name: 'Close' }));
    expect(onClose).toHaveBeenCalledTimes(2);
  });
});
