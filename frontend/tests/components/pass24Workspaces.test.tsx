import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import WorkspaceBar from '../../src/components/corpus/WorkspaceBar';

const noop = async () => undefined;

describe('pass 24: the workspace menu for a named key', () => {
  it('offers Default and New only to keys that may use them', () => {
    const { unmount } = render(
      <WorkspaceBar current="legal" workspaces={['legal']} onSwitch={() => undefined} onCreate={noop} onDelete={noop} />,
    );
    expect(screen.getAllByRole('option').map((option) => option.textContent)).toEqual(['Default', 'legal']);
    expect(screen.getByTestId('workspace-new')).toBeTruthy();
    unmount();
    render(
      <WorkspaceBar
        current="legal"
        workspaces={['legal']}
        onSwitch={() => undefined}
        onCreate={noop}
        onDelete={noop}
        defaultAllowed={false}
        canCreate={false}
      />,
    );
    expect(screen.getAllByRole('option').map((option) => option.textContent)).toEqual(['legal']);
    expect(screen.queryByTestId('workspace-new')).toBeNull();
  });
});
