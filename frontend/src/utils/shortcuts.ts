/** Whether the single-key shortcuts (?, [, ], j, k, 1-9) are on: a speech-input or
 *  switch user can set off a key by accident (WCAG 2.1.4), so they can be turned off.
 *  The ⌘ ones stay either way. On by default; remembered in this browser. */
export const SINGLE_KEY_SHORTCUTS_KEY = 'docrag-single-key-shortcuts';

export function singleKeyShortcutsOn(): boolean {
  try {
    return localStorage.getItem(SINGLE_KEY_SHORTCUTS_KEY) !== 'off';
  } catch {
    return true;
  }
}

export function setSingleKeyShortcuts(on: boolean): void {
  try {
    if (on) localStorage.removeItem(SINGLE_KEY_SHORTCUTS_KEY);
    else localStorage.setItem(SINGLE_KEY_SHORTCUTS_KEY, 'off');
  } catch {
    // Private mode: on for this session.
  }
}
