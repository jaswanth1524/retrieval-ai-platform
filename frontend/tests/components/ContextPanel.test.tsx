import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import ContextPanel from '../../src/components/ContextPanel';

function setup(overrides: Partial<Parameters<typeof ContextPanel>[0]> = {}) {
  const props = {
    title: 'Conversations',
    actionLabel: 'New',
    onAction: vi.fn(),
    apiStatus: 'ok' as const,
    chunkTotal: 42,
    children: <div>body</div>,
    ...overrides,
  };
  render(<ContextPanel {...props} />);
  return props;
}

describe('ContextPanel', () => {
  it('renders the title, action, children, and chunk total', () => {
    setup();
    expect(screen.getByText('Conversations')).toBeInTheDocument();
    expect(screen.getByText('body')).toBeInTheDocument();
    expect(screen.getByTestId('context-panel-footer')).toHaveTextContent('42 chunks');
  });

  it('fires onAction when the header button is clicked', async () => {
    const props = setup();
    await userEvent.click(screen.getByTestId('context-panel-action'));
    expect(props.onAction).toHaveBeenCalledOnce();
  });

  it('shows "api ok" when reachable and "api unreachable" on error', () => {
    const { rerender } = render(
      <ContextPanel title="x" actionLabel="y" onAction={vi.fn()} apiStatus="ok" chunkTotal={0}>
        <div />
      </ContextPanel>,
    );
    expect(screen.getByTestId('context-panel-footer')).toHaveTextContent('api ok');

    rerender(
      <ContextPanel title="x" actionLabel="y" onAction={vi.fn()} apiStatus="error" chunkTotal={0}>
        <div />
      </ContextPanel>,
    );
    expect(screen.getByTestId('context-panel-footer')).toHaveTextContent('api unreachable');
  });

  it('distinguishes an unauthorized server from an unreachable one', () => {
    setup({ apiStatus: 'unauthorized' });

    const footer = screen.getByTestId('context-panel-footer');
    expect(footer).toHaveTextContent('api key required');
    expect(footer).not.toHaveTextContent('api unreachable');
  });
});
