import { act, renderHook } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { useWindowFileDrop } from '../../src/hooks/useWindowFileDrop';

function dragEvent(type: string, types: string[], files: File[] = []): Event {
  const event = new Event(type, { bubbles: true, cancelable: true });
  Object.defineProperty(event, 'dataTransfer', {
    value: { types, files, dropEffect: 'none' },
  });
  return event;
}

describe('useWindowFileDrop', () => {
  it('uploads files dropped anywhere and stops the browser opening them', () => {
    const onFiles = vi.fn();
    const { result } = renderHook(() => useWindowFileDrop(true, onFiles));
    const file = new File(['x'], 'guide.pdf');

    act(() => {
      window.dispatchEvent(dragEvent('dragenter', ['Files']));
    });
    expect(result.current).toBe(true);
    const over = dragEvent('dragover', ['Files']);
    window.dispatchEvent(over);
    expect(over.defaultPrevented).toBe(true);

    const drop = dragEvent('drop', ['Files'], [file]);
    act(() => {
      window.dispatchEvent(drop);
    });

    expect(drop.defaultPrevented).toBe(true);
    expect(onFiles).toHaveBeenCalledWith([file]);
    expect(result.current).toBe(false);
  });

  it('leaves drags without files, and drops the corpus dropzone handled, alone', () => {
    const onFiles = vi.fn();
    renderHook(() => useWindowFileDrop(true, onFiles));

    const text = dragEvent('dragover', ['text/plain']);
    window.dispatchEvent(text);
    expect(text.defaultPrevented).toBe(false);

    const handled = dragEvent('drop', ['Files'], [new File(['x'], 'a.pdf')]);
    handled.preventDefault(); // what the dropzone's own handler does first
    window.dispatchEvent(handled);
    expect(onFiles).not.toHaveBeenCalled();
  });

  it('still blocks navigation but uploads nothing while disabled', () => {
    const onFiles = vi.fn();
    renderHook(() => useWindowFileDrop(false, onFiles));

    const over = dragEvent('dragover', ['Files']);
    window.dispatchEvent(over);
    const drop = dragEvent('drop', ['Files'], [new File(['x'], 'a.pdf')]);
    window.dispatchEvent(drop);

    expect(over.defaultPrevented).toBe(true);
    expect(drop.defaultPrevented).toBe(true);
    expect(onFiles).not.toHaveBeenCalled();
  });
});
