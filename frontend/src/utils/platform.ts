function isApplePlatform(): boolean {
  if (typeof navigator === 'undefined') return true;
  // Either source saying Apple counts: a browser with a spoofed user agent can report
  // different platforms in the two, and the Mac's own ⌘ must keep working.
  const platforms = [
    (navigator as Navigator & { userAgentData?: { platform?: string } }).userAgentData?.platform,
    navigator.platform,
  ];
  return platforms.some((platform) => /mac|iphone|ipad|ipod/i.test(platform ?? ''));
}

const APPLE = isApplePlatform();

/** A shortcut written in Mac notation ("⌘⇧O"), as this platform labels it ("Ctrl+Shift+O"
 *  off Apple devices). */
export function shortcutLabel(shortcut: string): string {
  if (APPLE) return shortcut;
  return shortcut.replace('⌘', 'Ctrl+').replace('⇧', 'Shift+');
}

/** The shortcut modifier: ⌘ on Apple devices, Ctrl elsewhere. On a Mac, Ctrl+K and
 *  friends are text-field editing keys (Ctrl+K deletes to the end of the line), so
 *  taking them for the app broke editing in every input. */
export function isShortcutModifier(event: KeyboardEvent, apple = APPLE): boolean {
  return apple ? event.metaKey : event.ctrlKey;
}
