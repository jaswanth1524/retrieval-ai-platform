import { describe, expect, it } from 'vitest';
import { formatRelativeTime, formatRelativeTimeAgo } from '../../src/utils/relativeTime';

const NOW = new Date('2026-08-05T12:00:00Z').getTime();

describe('formatRelativeTime', () => {
  it('returns "just now" for under a minute', () => {
    expect(formatRelativeTime(NOW - 30_000, NOW)).toBe('just now');
  });

  it('returns minutes under an hour', () => {
    expect(formatRelativeTime(NOW - 12 * 60_000, NOW)).toBe('12m');
  });

  it('returns hours under a day', () => {
    expect(formatRelativeTime(NOW - 2 * 3_600_000, NOW)).toBe('2h');
  });

  it('returns "yesterday" for exactly one day', () => {
    expect(formatRelativeTime(NOW - 24 * 3_600_000, NOW)).toBe('yesterday');
  });

  it('returns days under a week', () => {
    expect(formatRelativeTime(NOW - 3 * 24 * 3_600_000, NOW)).toBe('3d');
  });

  it('returns an absolute date beyond a week', () => {
    const result = formatRelativeTime(NOW - 10 * 24 * 3_600_000, NOW);
    expect(result).not.toMatch(/^\d+[mhd]$/);
    expect(result).not.toBe('yesterday');
  });
});

describe('formatRelativeTimeAgo', () => {
  it('appends "ago" to minute/hour/day forms', () => {
    expect(formatRelativeTimeAgo(NOW - 12 * 60_000, NOW)).toBe('12m ago');
    expect(formatRelativeTimeAgo(NOW - 2 * 3_600_000, NOW)).toBe('2h ago');
    expect(formatRelativeTimeAgo(NOW - 3 * 24 * 3_600_000, NOW)).toBe('3d ago');
  });

  it('does not append "ago" to "just now" or "yesterday"', () => {
    expect(formatRelativeTimeAgo(NOW - 30_000, NOW)).toBe('just now');
    expect(formatRelativeTimeAgo(NOW - 24 * 3_600_000, NOW)).toBe('yesterday');
  });
});
