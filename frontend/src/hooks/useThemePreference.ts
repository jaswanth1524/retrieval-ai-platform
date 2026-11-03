import { useEffect, useState } from 'react';
import { THEME_STORAGE_KEY } from '../appHelpers';
import { readStored, writeStored } from '../utils/safeStorage';
import { useStableHandler } from './useStableHandler';

/** The page's theme (on ``<html data-theme>``) and the rail's toggle for it. */
export function useThemePreference() {
  const [theme, setTheme] = useState<'dark' | 'light'>(
    () => (document.documentElement.dataset.theme === 'light' ? 'light' : 'dark'),
  );

  // Until a theme is picked the page follows the OS — live, not only at load: switching
  // the OS to dark at sunset left an open tab light. A pick made in another tab applies
  // here too (each tab used to keep its own).
  useEffect(() => {
    const apply = (next: 'dark' | 'light') => {
      setTheme(next);
      document.documentElement.dataset.theme = next;
    };
    const media = window.matchMedia?.('(prefers-color-scheme: light)');
    const onSystemChange = (event: MediaQueryListEvent) => {
      if (readStored(THEME_STORAGE_KEY) === null) apply(event.matches ? 'light' : 'dark');
    };
    const onStorage = (event: StorageEvent) => {
      if (event.key !== THEME_STORAGE_KEY) return;
      if (event.newValue === 'light' || event.newValue === 'dark') apply(event.newValue);
      else if (event.newValue === null && media) apply(media.matches ? 'light' : 'dark');
    };
    media?.addEventListener?.('change', onSystemChange);
    window.addEventListener('storage', onStorage);
    return () => {
      media?.removeEventListener?.('change', onSystemChange);
      window.removeEventListener('storage', onStorage);
    };
  }, []);

  const toggleTheme = useStableHandler(() => {
    const next = theme === 'dark' ? 'light' : 'dark';
    setTheme(next);
    document.documentElement.dataset.theme = next;
    writeStored(THEME_STORAGE_KEY, next);
  });

  return { theme, toggleTheme };
}
