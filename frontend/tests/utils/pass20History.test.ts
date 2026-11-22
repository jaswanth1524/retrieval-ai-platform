import { afterEach, describe, expect, it } from 'vitest';
import {
  REMIND_EVERY_MS,
  noteHistoryExported,
  takeHistoryReminder,
} from '../../src/utils/historyBackup';

afterEach(() => localStorage.clear());

describe('pass 20: history backup reminder', () => {
  it('reminds someone with history to lose, once a period, and not after an export', () => {
    const now = 1_000_000_000_000;
    expect(takeHistoryReminder(3, now)).toBe(false);
    expect(takeHistoryReminder(12, now)).toBe(true);
    expect(takeHistoryReminder(12, now + 1000)).toBe(false);
    expect(takeHistoryReminder(12, now + REMIND_EVERY_MS + 1)).toBe(true);
    noteHistoryExported(now + 3 * REMIND_EVERY_MS);
    expect(takeHistoryReminder(12, now + 3 * REMIND_EVERY_MS + 1000)).toBe(false);
  });
});
