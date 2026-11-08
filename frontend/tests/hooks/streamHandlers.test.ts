import { describe, expect, it, vi } from 'vitest';
import type { ChatTurn } from '../../src/components/ChatMessage';
import { answerStreamHandlers, createDeltaBuffer } from '../../src/hooks/streamHandlers';

function manualFrames() {
  const queued = new Map<number, () => void>();
  let next = 1;
  return {
    schedule: (flush: () => void) => {
      queued.set(next, flush);
      return next++;
    },
    unschedule: (id: number) => queued.delete(id),
    runAll: () => {
      const callbacks = [...queued.values()];
      queued.clear();
      callbacks.forEach((callback) => callback());
    },
    pending: () => queued.size,
  };
}

const blank: ChatTurn = {
  id: 'a',
  role: 'assistant',
  content: '',
  sources: [],
  timestamp: 0,
  timings: null,
  traceId: null,
};

describe('streamHandlers', () => {
  it('coalesces deltas into one update per frame and flushes on demand', () => {
    const frames = manualFrames();
    const applied: string[] = [];
    const pending = { current: 'stale' };
    const buffer = createDeltaBuffer(pending, (text) => applied.push(text), frames.schedule, frames.unschedule);
    expect(pending.current).toBe('');

    buffer.push('Hel');
    buffer.push('lo');
    expect(frames.pending()).toBe(1);
    expect(pending.current).toBe('Hello');
    frames.runAll();
    expect(applied).toEqual(['Hello']);

    buffer.push(' world');
    buffer.flush();
    expect(applied).toEqual(['Hello', ' world']);
    expect(frames.pending()).toBe(0);
    buffer.flush();
    expect(applied).toHaveLength(2);
  });

  it('can show the first text at once and coalesce the rest', () => {
    const frames = manualFrames();
    const applied: string[] = [];
    const buffer = createDeltaBuffer(
      { current: '' },
      (text) => applied.push(text),
      frames.schedule,
      frames.unschedule,
      true,
    );
    buffer.push('First');
    expect(applied).toEqual(['First']);
    expect(frames.pending()).toBe(0);
    buffer.push(' a');
    buffer.push(' b');
    expect(applied).toEqual(['First']);
    frames.runAll();
    expect(applied).toEqual(['First', ' a b']);
  });

  it('labels stages, writes sources onto the answer and flushes before done', () => {
    const frames = manualFrames();
    const stages: string[] = [];
    let turn = blank;
    const done = vi.fn();
    const followups: string[][] = [];
    const buffer = createDeltaBuffer(
      { current: '' },
      (text) => (turn = { ...turn, content: turn.content + text }),
      frames.schedule,
      frames.unschedule,
    );
    const handlers = answerStreamHandlers(true, buffer, {
      enterStage: (label) => stages.push(label),
      stageLabel: (stage, progress) => `${stage}${progress ? ` ${progress.step}/${progress.steps}` : ''}`,
      updateAnswer: (update) => (turn = update(turn)),
      setFollowups: (questions) => followups.push(questions),
      done,
    });

    handlers.onStage?.('summarizing', { step: 1, steps: 3 });
    handlers.onSources?.([], 'trace-1');
    handlers.onDelta?.('Part ');
    handlers.onDelta?.('one.');
    expect(turn.content).toBe('');
    handlers.onDone?.('Part one.', [], null, 'trace-1');

    expect(stages).toEqual(['summarizing 1/3', 'writing the summary…', 'writing the summary…']);
    expect(turn.traceId).toBe('trace-1');
    expect(turn.content).toBe('Part one.');
    expect(done).toHaveBeenCalledWith('Part one.', [], null, 'trace-1');

    handlers.onFollowups?.(['', '  ']);
    handlers.onFollowups?.(['What next?']);
    expect(followups).toEqual([['What next?']]);
  });
});
