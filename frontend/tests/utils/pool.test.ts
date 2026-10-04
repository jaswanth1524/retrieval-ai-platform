import { describe, expect, it } from 'vitest';
import { mapWithLimit } from '../../src/utils/pool';

describe('mapWithLimit', () => {
  it('never runs more than the limit at once and settles every item in order', async () => {
    let running = 0;
    let peak = 0;
    const results = await mapWithLimit([1, 2, 3, 4, 5], 2, async (n) => {
      running += 1;
      peak = Math.max(peak, running);
      await new Promise((resolve) => setTimeout(resolve, 5));
      running -= 1;
      if (n === 3) throw new Error('three');
      return n * 10;
    });

    expect(peak).toBe(2);
    expect(results.map((r) => (r.status === 'fulfilled' ? r.value : 'rejected'))).toEqual([
      10, 20, 'rejected', 40, 50,
    ]);
  });

  it('handles an empty list', async () => {
    expect(await mapWithLimit([], 2, async () => 1)).toEqual([]);
  });
});
