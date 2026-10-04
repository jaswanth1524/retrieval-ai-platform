/** localStorage that never throws.
 *
 * Storage can be denied or full (private windows, quota, site-data blocked). Every
 * preference stored here is best-effort: the in-session value still applies when
 * reading or writing fails, so a failure is swallowed rather than breaking the app. */

export function readStored(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function writeStored(key: string, value: string | null): void {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {
    // Best-effort, see above.
  }
}

/** Parse a stored JSON value; null when it is absent, unreadable or not JSON. */
export function readStoredJson<T>(key: string): T | null {
  const raw = readStored(key);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}
