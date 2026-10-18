import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import WorkspaceBar, { WORKSPACE_BROWSE_COMMIT_MS } from '../../src/components/corpus/WorkspaceBar';

function setup(overrides: Partial<Parameters<typeof WorkspaceBar>[0]> = {}) {
  const props = {
    current: '',
    workspaces: ['legal', 'research'],
    onSwitch: vi.fn(),
    onCreate: vi.fn().mockResolvedValue(undefined),
    onDelete: vi.fn().mockResolvedValue(undefined),
    ...overrides,
  };
  render(<WorkspaceBar {...props} />);
  return props;
}

describe('WorkspaceBar', () => {
  it('switches between the default and the named workspaces', async () => {
    const props = setup();
    const select = screen.getByLabelText('Workspace');
    expect(screen.getAllByRole('option').map((option) => option.textContent)).toEqual([
      'Default',
      'legal',
      'research',
    ]);
    await userEvent.selectOptions(select, 'legal');
    expect(props.onSwitch).toHaveBeenCalledWith('legal');
    // The default workspace can't be deleted.
    expect(screen.queryByTestId('workspace-delete')).not.toBeInTheDocument();
  });

  it('switches once after arrowing through the closed select, not on every option', () => {
    vi.useFakeTimers();
    try {
      const props = setup();
      const select = screen.getByLabelText<HTMLSelectElement>('Workspace');
      // What Windows and Linux browsers do: each arrow press changes the value.
      fireEvent.keyDown(select, { key: 'ArrowDown' });
      fireEvent.change(select, { target: { value: 'legal' } });
      expect(select.value).toBe('legal'); // shown while it waits
      fireEvent.keyDown(select, { key: 'ArrowDown' });
      fireEvent.change(select, { target: { value: 'research' } });
      expect(props.onSwitch).not.toHaveBeenCalled();
      act(() => vi.advanceTimersByTime(WORKSPACE_BROWSE_COMMIT_MS));
      expect(props.onSwitch).toHaveBeenCalledTimes(1);
      expect(props.onSwitch).toHaveBeenCalledWith('research');
    } finally {
      vi.useRealTimers();
    }
  });

  it('switches to a keyboard pick at once on Enter or when focus leaves', () => {
    const props = setup();
    const select = screen.getByLabelText<HTMLSelectElement>('Workspace');
    fireEvent.keyDown(select, { key: 'ArrowDown' });
    fireEvent.change(select, { target: { value: 'legal' } });
    fireEvent.keyDown(select, { key: 'Enter' });
    expect(props.onSwitch).toHaveBeenCalledWith('legal');

    fireEvent.keyDown(select, { key: 'r' });
    fireEvent.change(select, { target: { value: 'research' } });
    fireEvent.blur(select);
    expect(props.onSwitch).toHaveBeenLastCalledWith('research');
    expect(props.onSwitch).toHaveBeenCalledTimes(2);
  });

  it('creates a workspace, checking the name first and showing the server refusal', async () => {
    const onCreate = vi
      .fn()
      .mockRejectedValueOnce(new Error("Workspace 'legal' already exists."))
      .mockResolvedValueOnce(undefined);
    setup({ onCreate });

    await userEvent.click(screen.getByTestId('workspace-new'));
    const input = screen.getByLabelText('New workspace name');
    await userEvent.type(input, '-bad{Enter}');
    expect(screen.getByRole('alert')).toHaveTextContent('lowercase letters');
    expect(onCreate).not.toHaveBeenCalled();

    await userEvent.clear(input);
    await userEvent.type(input, 'Legal{Enter}');
    expect(onCreate).toHaveBeenCalledWith('legal');
    expect(await screen.findByRole('alert')).toHaveTextContent('already exists');

    await userEvent.click(screen.getByRole('button', { name: 'Create' }));
    expect(onCreate).toHaveBeenCalledTimes(2);
    expect(screen.queryByLabelText('New workspace name')).not.toBeInTheDocument();
  });

  it('deletes the current workspace after a confirm', async () => {
    const props = setup({ current: 'legal' });
    await userEvent.click(screen.getByTestId('workspace-delete'));
    await userEvent.click(screen.getByRole('button', { name: 'Delete' }));
    expect(props.onDelete).toHaveBeenCalledWith('legal');
  });

  it('only switches for the read-only key', () => {
    setup({ current: 'legal', readOnly: true });
    expect(screen.getByLabelText('Workspace')).toBeEnabled();
    expect(screen.queryByTestId('workspace-new')).not.toBeInTheDocument();
    expect(screen.queryByTestId('workspace-delete')).not.toBeInTheDocument();
  });

  it('always shows the workspace in use as selected, even when the list lacks it', () => {
    // Created while the list couldn't be read, or deleted elsewhere: the select used to
    // show "Default" while requests went to this one.
    setup({ current: 'fresh', workspaces: ['legal'] });
    expect(screen.getByLabelText('Workspace')).toHaveValue('fresh');
    expect(screen.getByRole('option', { name: 'fresh' })).toBeInTheDocument();
  });

  it('names the workspace on New and Delete', () => {
    setup({ current: 'legal' });
    expect(screen.getByRole('button', { name: 'New workspace' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Delete workspace legal' })).toBeInTheDocument();
  });

  it('returns focus to New after Cancel, and Escape cancels without reaching the drawer', async () => {
    const outerEscape = vi.fn();
    render(
      <div onKeyDown={(event) => event.key === 'Escape' && outerEscape()}>
        <WorkspaceBar
          current=""
          workspaces={[]}
          onSwitch={vi.fn()}
          onCreate={vi.fn()}
          onDelete={vi.fn()}
        />
      </div>,
    );
    await userEvent.click(screen.getByTestId('workspace-new'));
    await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(screen.getByTestId('workspace-new')).toHaveFocus();

    await userEvent.click(screen.getByTestId('workspace-new'));
    expect(screen.getByLabelText('New workspace name')).toHaveFocus();
    await userEvent.keyboard('{Escape}');
    expect(screen.queryByLabelText('New workspace name')).not.toBeInTheDocument();
    expect(outerEscape).not.toHaveBeenCalled();
    expect(screen.getByTestId('workspace-new')).toHaveFocus();
  });

  it('focuses the select after a create or a confirmed delete', async () => {
    const props = setup({ current: 'legal' });
    await userEvent.click(screen.getByTestId('workspace-delete'));
    await userEvent.click(screen.getByRole('button', { name: 'Delete' }));
    await waitFor(() => expect(screen.getByLabelText('Workspace')).toHaveFocus());
    expect(props.onDelete).toHaveBeenCalledWith('legal');

    await userEvent.click(screen.getByTestId('workspace-new'));
    await userEvent.type(screen.getByLabelText('New workspace name'), 'notes{Enter}');
    await waitFor(() => expect(screen.getByLabelText('Workspace')).toHaveFocus());
  });

  it('ties its error to the field it is about', async () => {
    setup();
    await userEvent.click(screen.getByTestId('workspace-new'));
    const input = screen.getByLabelText('New workspace name');
    await userEvent.type(input, '-bad{Enter}');
    expect(input).toHaveAccessibleDescription('Use 1–32 lowercase letters, digits or hyphens.');
    expect(input).toHaveAttribute('aria-invalid', 'true');
  });
});
