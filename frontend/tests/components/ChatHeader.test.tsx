import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import ChatHeader from '../../src/components/ChatHeader';
import type { ChatTurn } from '../../src/components/ChatMessage';

function makeTurn(overrides: Partial<ChatTurn> = {}): ChatTurn {
  return {
    id: 't1',
    role: 'user',
    content: 'Hi',
    sources: [],
    timestamp: 0,
    timings: null,
    traceId: null,
    ...overrides,
  };
}

function baseProps(overrides: Partial<Parameters<typeof ChatHeader>[0]> = {}) {
  return {
    title: 'Contract liability caps',
    scopeLabel: 'All 4 documents',
    mode: 'reader' as const,
    onSetMode: vi.fn(),
    turns: [] as ChatTurn[],
    onClear: vi.fn(),
    onOpenPalette: vi.fn(),
    inspectorOpen: false,
    onToggleInspector: vi.fn(),
    ...overrides,
  };
}

describe('ChatHeader', () => {
  it('renders the title and scope label', () => {
    render(<ChatHeader {...baseProps()} />);
    expect(screen.getByText('Contract liability caps')).toBeInTheDocument();
    expect(screen.getByText('All 4 documents')).toBeInTheDocument();
  });

  it('hides Clear/Export when there are no turns', () => {
    render(<ChatHeader {...baseProps({ turns: [] })} />);
    expect(screen.queryByTestId('chat-header-clear')).not.toBeInTheDocument();
    expect(screen.queryByTestId('chat-header-export')).not.toBeInTheDocument();
  });

  it('shows Clear/Export once there are turns', () => {
    render(<ChatHeader {...baseProps({ turns: [makeTurn()] })} />);
    expect(screen.getByTestId('chat-header-clear')).toBeInTheDocument();
    expect(screen.getByTestId('chat-header-export')).toBeInTheDocument();
  });

  it('requires confirmation before firing onClear', async () => {
    const props = baseProps({ turns: [makeTurn()] });
    render(<ChatHeader {...props} />);

    await userEvent.click(screen.getByTestId('chat-header-clear'));
    expect(props.onClear).not.toHaveBeenCalled();

    await userEvent.click(screen.getByTestId('chat-header-clear-confirm'));
    expect(props.onClear).toHaveBeenCalledOnce();
  });

  it('marks the active mode with aria-checked and switches on click', async () => {
    const props = baseProps();
    render(<ChatHeader {...props} />);

    expect(screen.getByTestId('mode-reader')).toHaveAttribute('aria-checked', 'true');
    expect(screen.getByTestId('mode-engineer')).toHaveAttribute('aria-checked', 'false');

    await userEvent.click(screen.getByTestId('mode-engineer'));
    expect(props.onSetMode).toHaveBeenCalledWith('engineer');
  });

  it('fires onOpenPalette and onToggleInspector', async () => {
    const props = baseProps();
    render(<ChatHeader {...props} />);

    await userEvent.click(screen.getByTestId('open-palette'));
    expect(props.onOpenPalette).toHaveBeenCalledOnce();

    await userEvent.click(screen.getByTestId('toggle-inspector'));
    expect(props.onToggleInspector).toHaveBeenCalledOnce();
  });
});
