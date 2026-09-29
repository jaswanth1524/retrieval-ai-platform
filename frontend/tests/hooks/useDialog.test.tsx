import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { useDialog } from '../../src/hooks/useDialog';

function Dialog({ name, open, onClose }: { name: string; open: boolean; onClose: () => void }) {
  const ref = useDialog(open, onClose);
  if (!open) return null;
  return (
    <div ref={ref} data-testid={name}>
      <button type="button">{name} first</button>
      <button type="button">{name} last</button>
    </div>
  );
}

describe('useDialog', () => {
  it('lets only the innermost of two open dialogs handle Escape and Tab', async () => {
    const closeOuter = vi.fn();
    const closeInner = vi.fn();
    const { rerender } = render(
      <>
        <Dialog name="outer" open onClose={closeOuter} />
        <Dialog name="inner" open={false} onClose={closeInner} />
      </>,
    );
    // Opening the inner dialog after the outer mirrors the document viewer opened
    // from the inspector drawer.
    rerender(
      <>
        <Dialog name="outer" open onClose={closeOuter} />
        <Dialog name="inner" open onClose={closeInner} />
      </>,
    );
    expect(screen.getByText('inner first')).toHaveFocus();

    // Tab wraps inside the inner dialog instead of being pulled into the outer one.
    await userEvent.tab();
    expect(screen.getByText('inner last')).toHaveFocus();
    await userEvent.tab();
    expect(screen.getByText('inner first')).toHaveFocus();

    await userEvent.keyboard('{Escape}');
    expect(closeInner).toHaveBeenCalledTimes(1);
    expect(closeOuter).not.toHaveBeenCalled();

    // Once the inner one is gone, the outer handles keys again.
    rerender(
      <>
        <Dialog name="outer" open onClose={closeOuter} />
        <Dialog name="inner" open={false} onClose={closeInner} />
      </>,
    );
    await userEvent.keyboard('{Escape}');
    expect(closeOuter).toHaveBeenCalledTimes(1);
  });
});
