import { renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useLinks } from '../../src/hooks/useLinks';

afterEach(() => {
  window.history.replaceState(null, '', '/');
});

describe('pass 21: a passage link opened before the server answers', () => {
  it('waits for it, then opens the passage', () => {
    window.history.replaceState(null, '', '/#d=guide.md');
    const setSourceView = vi.fn();
    const options = {
      historyReady: true,
      serverReady: false,
      conversations: [],
      activeConversationId: null,
      workspace: '',
      sourceView: null,
      setSourceView,
      setPanelDrawerOpen: vi.fn(),
      chooseWorkspaceRef: { current: vi.fn() },
      switchConversationFromList: vi.fn(),
      pushToast: vi.fn(),
    };
    const { rerender } = renderHook((props) => useLinks(props), { initialProps: options });
    expect(setSourceView).not.toHaveBeenCalled();
    expect(window.location.hash).toBe('#d=guide.md');
    rerender({ ...options, serverReady: true });
    expect(setSourceView).toHaveBeenCalledWith({ filename: 'guide.md', chunkId: null });
    expect(window.location.hash).toBe('');
  });
});
