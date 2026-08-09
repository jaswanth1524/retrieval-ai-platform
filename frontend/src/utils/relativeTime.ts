const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

/** Compact relative time for list rows ("12m", "2h", "3d ago" territory) — mirrors the
 * design's conversation and trace row meta lines, which use "12m ago" / "2h" / "yesterday". */
export function formatRelativeTime(timestampMs: number, now: number = Date.now()): string {
  const diff = Math.max(0, now - timestampMs);
  if (diff < MINUTE) return 'just now';
  if (diff < HOUR) return `${Math.floor(diff / MINUTE)}m`;
  if (diff < DAY) return `${Math.floor(diff / HOUR)}h`;
  const days = Math.floor(diff / DAY);
  if (days === 1) return 'yesterday';
  if (days < 7) return `${days}d`;
  return new Date(timestampMs).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

/** Same as formatRelativeTime, with a trailing "ago" where that still reads naturally
 * ("12m ago", "3d ago") — not appended to "just now", "yesterday", or an absolute date. */
export function formatRelativeTimeAgo(timestampMs: number, now: number = Date.now()): string {
  const base = formatRelativeTime(timestampMs, now);
  return /^\d+[mhd]$/.test(base) ? `${base} ago` : base;
}
