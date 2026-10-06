import { renderHook } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { type CommandActions, type CommandState, useCommands } from '../../src/hooks/useCommands';
import { isShortcutModifier } from '../../src/utils/platform';

function state(overrides: Partial<CommandState> = {}): CommandState {
  return {
    pending: false,
    configLoaded: true,
    inspectorOpen: false,
    mode: 'reader',
    theme: 'dark',
    hasTurns: true,
    conversations: [],
    activeConversationId: null,
    readOnly: false,
    restoring: false,
    exportingBackup: false,
    ...overrides,
  };
}

function actions(): CommandActions {
  return {
    newConversation: vi.fn(),
    upload: vi.fn(),
    openSettings: vi.fn(),
    toggleInspector: vi.fn(),
    setMode: vi.fn(),
    toggleTheme: vi.fn(),
    exportMarkdown: vi.fn(),
    exportJson: vi.fn(),
    browseTraces: vi.fn(),
    searchPassages: vi.fn(),
    restoreBackup: vi.fn(),
    downloadBackup: vi.fn(),
    importConversation: vi.fn(),
    clear: vi.fn(),
    switchConversation: vi.fn(),
  };
}

function press(key: string, init: KeyboardEventInit = {}) {
  // jsdom reports a non-Apple platform, where the shortcut modifier is Ctrl.
  document.dispatchEvent(new KeyboardEvent('keydown', { key, ctrlKey: true, ...init }));
}

describe('useCommands', () => {
  it('runs the command a shortcut names, and closes the palette first', () => {
    const run = actions();
    const palette = { otherDialogOpen: false, togglePalette: vi.fn(), closePalette: vi.fn() };
    renderHook(() => useCommands(state(), run, palette));

    press('u');
    press('o', { shiftKey: true });
    press('k');

    expect(run.upload).toHaveBeenCalledTimes(1);
    expect(run.newConversation).toHaveBeenCalledTimes(1);
    expect(palette.closePalette).toHaveBeenCalledTimes(2);
    expect(palette.togglePalette).toHaveBeenCalledTimes(1);
  });

  it('runs the latest actions, without rebuilding the list for new closures', () => {
    const first = actions();
    const second = actions();
    const palette = { otherDialogOpen: false, togglePalette: vi.fn(), closePalette: vi.fn() };
    const fixed = state();
    const { result, rerender } = renderHook(
      ({ run }: { run: CommandActions }) => useCommands(fixed, run, palette),
      { initialProps: { run: first } },
    );
    const list = result.current;

    rerender({ run: second });
    result.current.find((command) => command.id === 'toggle-theme')!.run();

    expect(result.current).toBe(list);
    expect(second.toggleTheme).toHaveBeenCalledTimes(1);
    expect(first.toggleTheme).not.toHaveBeenCalled();
  });

  it('binds nothing over another dialog, and skips disabled commands', () => {
    const run = actions();
    const palette = { otherDialogOpen: true, togglePalette: vi.fn(), closePalette: vi.fn() };
    const { rerender } = renderHook(
      ({ open, pending }: { open: boolean; pending: boolean }) =>
        useCommands(state({ pending }), run, { ...palette, otherDialogOpen: open }),
      { initialProps: { open: true, pending: false } },
    );

    press('k');
    press('u');
    expect(palette.togglePalette).not.toHaveBeenCalled();
    expect(run.upload).not.toHaveBeenCalled();

    rerender({ open: false, pending: true });
    press('o', { shiftKey: true });
    expect(run.newConversation).not.toHaveBeenCalled();
  });

  it('leaves backup and restore out for a read-only key', () => {
    const { result } = renderHook(() =>
      useCommands(state({ readOnly: true }), actions(), {
        otherDialogOpen: false,
        togglePalette: vi.fn(),
        closePalette: vi.fn(),
      }),
    );

    const ids = result.current.map((command) => command.id);
    expect(ids).not.toContain('restore-backup');
    expect(ids).not.toContain('export-corpus');
  });

  it('leaves upload out for a read-only key, shortcut included', () => {
    const run = actions();
    const { result } = renderHook(() =>
      useCommands(state({ readOnly: true }), run, {
        otherDialogOpen: false,
        togglePalette: vi.fn(),
        closePalette: vi.fn(),
      }),
    );

    press('u');

    expect(result.current.map((command) => command.id)).not.toContain('upload');
    expect(run.upload).not.toHaveBeenCalled();
  });
});

describe('isShortcutModifier', () => {
  it('takes ⌘ on a Mac, leaving Ctrl+K to text fields, and Ctrl elsewhere', () => {
    const ctrl = new KeyboardEvent('keydown', { key: 'k', ctrlKey: true });
    const meta = new KeyboardEvent('keydown', { key: 'k', metaKey: true });

    expect(isShortcutModifier(meta, true)).toBe(true);
    expect(isShortcutModifier(ctrl, true)).toBe(false);
    expect(isShortcutModifier(ctrl, false)).toBe(true);
    expect(isShortcutModifier(meta, false)).toBe(false);
  });
});
