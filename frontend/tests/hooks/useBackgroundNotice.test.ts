import { renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useBackgroundNotice } from '../../src/hooks/useBackgroundNotice';

function setHidden(hidden: boolean) {
  Object.defineProperty(document, 'hidden', { configurable: true, get: () => hidden });
}

afterEach(() => {
  setHidden(false);
  document.title = 'DocRAG';
  vi.unstubAllGlobals();
});

describe('useBackgroundNotice', () => {
  it('counts answers finished in a hidden tab in the title until the tab is shown', () => {
    document.title = 'DocRAG';
    const { rerender } = renderHook(({ pending }) => useBackgroundNotice(pending, false, 'q'), {
      initialProps: { pending: true },
    });
    setHidden(true);
    rerender({ pending: false });
    rerender({ pending: true });
    rerender({ pending: false });
    expect(document.title).toBe('(2) DocRAG');

    setHidden(false);
    document.dispatchEvent(new Event('visibilitychange'));
    expect(document.title).toBe('DocRAG');
  });

  it('leaves the title alone for an answer finished in the visible tab', () => {
    document.title = 'DocRAG';
    const { rerender } = renderHook(({ pending }) => useBackgroundNotice(pending, true, 'q'), {
      initialProps: { pending: true },
    });
    rerender({ pending: false });
    expect(document.title).toBe('DocRAG');
  });

  it('notifies only when turned on and permitted', () => {
    const shown: string[] = [];
    class FakeNotification {
      static permission = 'granted';
      constructor(title: string) {
        shown.push(title);
      }
    }
    vi.stubGlobal('Notification', FakeNotification);
    setHidden(true);
    const off = renderHook(({ pending }) => useBackgroundNotice(pending, false, 'q'), {
      initialProps: { pending: true },
    });
    off.rerender({ pending: false });
    const on = renderHook(({ pending }) => useBackgroundNotice(pending, true, 'q'), {
      initialProps: { pending: true },
    });
    on.rerender({ pending: false });
    expect(shown).toEqual(['Answer ready']);
  });

  it('says when the question failed or was stopped instead of "Answer ready"', () => {
    const shown: string[] = [];
    class FakeNotification {
      static permission = 'granted';
      constructor(title: string) {
        shown.push(title);
      }
    }
    vi.stubGlobal('Notification', FakeNotification);
    setHidden(true);
    for (const outcome of ['failed', 'stopped'] as const) {
      const hook = renderHook(
        ({ pending }) => useBackgroundNotice(pending, true, 'q', outcome),
        { initialProps: { pending: true } },
      );
      hook.rerender({ pending: false });
    }
    expect(shown).toEqual(['Question failed', 'Answer stopped']);
  });
});
