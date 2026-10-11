import { describe, expect, it, vi } from 'vitest';
import { createLimiter, mapWithLimit } from '../../src/utils/pool';

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

describe('createLimiter', () => {
  it('runs at most `limit` tasks at once across callers that start at different times', async () => {
    const limiter = createLimiter(2);
    const releases: Array<() => void> = [];
    const started: number[] = [];
    const task = (n: number) => () =>
      new Promise<number>((resolve) => {
        started.push(n);
        releases.push(() => resolve(n));
      });

    const first = limiter.run(task(1));
    const second = limiter.run(task(2));
    const third = limiter.run(task(3));
    await Promise.resolve();
    expect(started).toEqual([1, 2]);

    releases[0]();
    expect(await first).toBe(1);
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(started).toEqual([1, 2, 3]);
    releases[1]();
    releases[2]();
    expect(await Promise.all([second, third])).toEqual([2, 3]);
  });

  it('drops a waiting task whose signal aborts, without running it', async () => {
    const limiter = createLimiter(1);
    let release = () => {};
    const busy = limiter.run(() => new Promise<void>((resolve) => (release = resolve)));
    const controller = new AbortController();
    const ran = vi.fn(async () => 'ran');
    const waiting = limiter.run(ran, controller.signal);

    controller.abort();
    await expect(waiting).rejects.toBeDefined();
    release();
    await busy;
    expect(ran).not.toHaveBeenCalled();
    // The slot is free again.
    expect(await limiter.run(async () => 'next')).toBe('next');
  });
});
