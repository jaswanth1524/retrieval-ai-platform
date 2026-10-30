import type { UrlRefreshStatusResponse } from '../api/types';
import { formatRelativeTimeAgo } from './relativeTime';

/** The intervals offered for a fetched document's own refresh (minutes; null = the
 *  server's URL_REFRESH_INTERVAL_MINUTES, 0 = never). */
export const REFRESH_CHOICES: { label: string; minutes: number | null }[] = [
  { label: 'server default', minutes: null },
  { label: 'every 15 min', minutes: 15 },
  { label: 'hourly', minutes: 60 },
  { label: 'every 6 h', minutes: 360 },
  { label: 'daily', minutes: 1440 },
  { label: 'weekly', minutes: 10080 },
  { label: 'never', minutes: 0 },
];

function inFuture(seconds: number, now: number): string {
  const minutes = Math.round((seconds * 1000 - now) / 60_000);
  if (minutes <= 0) return 'due now';
  if (minutes < 60) return `next in ${minutes} min`;
  const hours = Math.round(minutes / 60);
  if (hours < 48) return `next in ${hours} h`;
  return `next in ${Math.round(hours / 24)} d`;
}

/** One line on a fetched document's card: when it was last checked and when next. */
export function refreshSummary(status: UrlRefreshStatusResponse, now: number = Date.now()): string {
  const parts: string[] = [];
  if (status.last_checked != null) {
    const outcome =
      status.last_outcome === 'failed'
        ? `check failed${(status.failures ?? 0) > 1 ? ` ${status.failures}×` : ''}`
        : status.last_outcome === 'queued'
          ? // A full fetch was indexed (the same bytes are still left as they were).
            'fetched'
          : 'unchanged';
    parts.push(`${outcome} ${formatRelativeTimeAgo(status.last_checked * 1000, now)}`);
  }
  if (status.refresh_minutes === 0) parts.push('not checked automatically');
  else if (status.next_due != null) parts.push(inFuture(status.next_due, now));
  return parts.join(' · ');
}
