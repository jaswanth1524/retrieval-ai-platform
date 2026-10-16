import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import WorkspaceBar from '../../src/components/corpus/WorkspaceBar';

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
});
