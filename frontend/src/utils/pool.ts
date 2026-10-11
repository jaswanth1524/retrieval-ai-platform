/** Run `task` over `items` with at most `limit` running at once; settles like
 *  Promise.allSettled, in input order. */
export async function mapWithLimit<T, R>(
  items: readonly T[],
  limit: number,
  task: (item: T, index: number) => Promise<R>,
): Promise<PromiseSettledResult<R>[]> {
  const results: PromiseSettledResult<R>[] = new Array(items.length);
  let next = 0;
  const worker = async () => {
    while (next < items.length) {
      const index = next;
      next += 1;
      try {
        results[index] = { status: 'fulfilled', value: await task(items[index], index) };
      } catch (reason) {
        results[index] = { status: 'rejected', reason };
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return results;
}

export interface Limiter {
  /** Runs `task` once fewer than `limit` tasks are running. Rejects with the signal's
   *  reason, without running the task, when `signal` aborts while it waits. */
  run<R>(task: () => Promise<R>, signal?: AbortSignal): Promise<R>;
}

/** A slot pool shared by callers that start at different times, unlike mapWithLimit,
 *  which runs one batch. */
export function createLimiter(limit: number): Limiter {
  let running = 0;
  const waiting: Array<() => void> = [];
  const release = () => {
    running -= 1;
    waiting.shift()?.();
  };
  return {
    async run(task, signal) {
      if (running >= limit) {
        await new Promise<void>((resolve, reject) => {
          const start = () => {
            signal?.removeEventListener('abort', onAbort);
            resolve();
          };
          const onAbort = () => {
            const index = waiting.indexOf(start);
            if (index >= 0) waiting.splice(index, 1);
            reject(signal?.reason ?? new DOMException('Aborted', 'AbortError'));
          };
          if (signal?.aborted) {
            onAbort();
            return;
          }
          signal?.addEventListener('abort', onAbort, { once: true });
          waiting.push(start);
        });
      }
      running += 1;
      try {
        return await task();
      } finally {
        release();
      }
    },
  };
}
