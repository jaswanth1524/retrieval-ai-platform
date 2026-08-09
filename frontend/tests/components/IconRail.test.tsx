import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import IconRail from '../../src/components/IconRail';

function setup(overrides: Partial<Parameters<typeof IconRail>[0]> = {}) {
  const props = {
    active: 'chat' as const,
    onSelect: vi.fn(),
    theme: 'dark' as const,
    onToggleTheme: vi.fn(),
    onOpenSettings: vi.fn(),
    ...overrides,
  };
  render(<IconRail {...props} />);
  return props;
}

describe('IconRail', () => {
  it('marks the active panel button as pressed', () => {
    setup({ active: 'corpus' });
    expect(screen.getByTestId('rail-corpus')).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByTestId('rail-chat')).toHaveAttribute('aria-pressed', 'false');
  });

  it('fires onSelect with the clicked panel key', async () => {
    const props = setup();
    await userEvent.click(screen.getByTestId('rail-traces'));
    expect(props.onSelect).toHaveBeenCalledWith('traces');
  });

  it('fires onToggleTheme and onOpenSettings', async () => {
    const props = setup();
    await userEvent.click(screen.getByTestId('theme-toggle'));
    expect(props.onToggleTheme).toHaveBeenCalledOnce();

    await userEvent.click(screen.getByTestId('open-settings'));
    expect(props.onOpenSettings).toHaveBeenCalledOnce();
  });

  it('labels the theme toggle for the opposite theme', () => {
    setup({ theme: 'dark' });
    expect(screen.getByLabelText('Switch to light theme')).toBeInTheDocument();
  });
});
