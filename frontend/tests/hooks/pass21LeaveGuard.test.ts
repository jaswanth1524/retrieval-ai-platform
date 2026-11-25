import { renderHook } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { useLeaveGuard } from '../../src/hooks/useLeaveGuard';

function leave(): boolean {
  const event = new Event('beforeunload', { cancelable: true });
  window.dispatchEvent(event);
  return event.defaultPrevented;
}

describe('pass 21: leaving mid-answer', () => {
  it('asks first only while active', () => {
    const { rerender, unmount } = renderHook(({ active }) => useLeaveGuard(active), {
      initialProps: { active: false },
    });
    expect(leave()).toBe(false);
    rerender({ active: true });
    expect(leave()).toBe(true);
    rerender({ active: false });
    expect(leave()).toBe(false);
    rerender({ active: true });
    unmount();
    expect(leave()).toBe(false);
  });
});
