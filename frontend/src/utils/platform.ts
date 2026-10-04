function isApplePlatform(): boolean {
  if (typeof navigator === 'undefined') return true;
  const platform =
    (navigator as Navigator & { userAgentData?: { platform?: string } }).userAgentData?.platform ??
    navigator.platform ??
    '';
  return /mac|iphone|ipad|ipod/i.test(platform);
}

const APPLE = isApplePlatform();

/** A shortcut written in Mac notation ("⌘⇧O"), as this platform labels it ("Ctrl+Shift+O"
 *  off Apple devices). The key handler accepts ⌘ or Ctrl everywhere; only the label
 *  differs. */
export function shortcutLabel(shortcut: string): string {
  if (APPLE) return shortcut;
  return shortcut.replace('⌘', 'Ctrl+').replace('⇧', 'Shift+');
}
