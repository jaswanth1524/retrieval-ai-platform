import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import ConfirmInline from '../../src/components/ConfirmInline';

describe('ConfirmInline', () => {
  it('focuses Cancel, the safe choice, and confirms on the other button', async () => {
    const onConfirm = vi.fn();
    render(
      <ConfirmInline prompt="Delete?" confirmLabel="Delete" onConfirm={onConfirm} onCancel={vi.fn()} />,
    );

    expect(screen.getByText('Delete?')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Cancel' })).toHaveFocus();
    await userEvent.click(screen.getByRole('button', { name: 'Delete' }));
    expect(onConfirm).toHaveBeenCalledTimes(1);
  });

  it('backs out on Escape without letting it reach the drawer around it', () => {
    const onCancel = vi.fn();
    const outer = vi.fn();
    render(
      <div onKeyDown={outer}>
        <ConfirmInline confirmLabel="Confirm" onConfirm={vi.fn()} onCancel={onCancel} />
      </div>,
    );

    fireEvent.keyDown(screen.getByRole('button', { name: 'Cancel' }), { key: 'Escape' });

    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(outer).not.toHaveBeenCalled();
  });

  it('shows a running action and blocks a second press, but Cancel stays usable', () => {
    const { rerender } = render(
      <ConfirmInline confirmLabel="Confirm" onConfirm={vi.fn()} onCancel={vi.fn()} busy confirmTestId="yes" />,
    );
    expect(screen.getByTestId('yes')).toHaveTextContent('…');
    expect(screen.getByTestId('yes')).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeEnabled();

    rerender(
      <ConfirmInline confirmLabel="Confirm" onConfirm={vi.fn()} onCancel={vi.fn()} disabled confirmTestId="yes" />,
    );
    expect(screen.getByTestId('yes')).toHaveTextContent('Confirm');
    expect(screen.getByTestId('yes')).toBeDisabled();
  });
});
