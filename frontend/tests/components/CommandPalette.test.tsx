import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import CommandPalette from '../../src/components/CommandPalette';
import type { Command } from '../../src/components/CommandPalette';

function makeCommands(overrides: Partial<Command>[] = []): Command[] {
  const base: Command[] = [
    { id: 'new', glyph: '＋', label: 'New conversation', shortcut: '⌘N', run: vi.fn() },
    { id: 'upload', glyph: '↑', label: 'Upload a document', run: vi.fn() },
    { id: 'theme', glyph: '☀', label: 'Use light theme', run: vi.fn() },
  ];
  return base.map((command, index) => ({ ...command, ...overrides[index] }));
}

function setup(commands = makeCommands()) {
  const onClose = vi.fn();
  render(<CommandPalette commands={commands} onClose={onClose} />);
  return { commands, onClose };
}

describe('CommandPalette', () => {
  it('lists every command and focuses the search input', () => {
    setup();

    expect(screen.getAllByTestId('palette-row')).toHaveLength(3);
    // Typing must land in the search box, not on the first result.
    expect(screen.getByTestId('palette-input')).toHaveFocus();
  });

  it('filters by substring', async () => {
    setup();

    await userEvent.type(screen.getByTestId('palette-input'), 'theme');

    const rows = screen.getAllByTestId('palette-row');
    expect(rows).toHaveLength(1);
    expect(rows[0]).toHaveTextContent('Use light theme');
  });

  it('shows an empty state when nothing matches', async () => {
    setup();

    await userEvent.type(screen.getByTestId('palette-input'), 'zzzz');

    expect(screen.getByTestId('palette-empty')).toBeInTheDocument();
  });

  it('runs the highlighted command on Enter and closes first', async () => {
    const { commands, onClose } = setup();

    await userEvent.keyboard('{Enter}');

    expect(commands[0].run).toHaveBeenCalled();
    expect(onClose).toHaveBeenCalled();
  });

  it('moves the highlight with arrow keys', async () => {
    const { commands } = setup();

    await userEvent.keyboard('{ArrowDown}{ArrowDown}{Enter}');

    expect(commands[2].run).toHaveBeenCalled();
    expect(commands[0].run).not.toHaveBeenCalled();
  });

  it('wraps the highlight around both ends', async () => {
    const { commands } = setup();

    // Up from the first entry lands on the last.
    await userEvent.keyboard('{ArrowUp}{Enter}');

    expect(commands[2].run).toHaveBeenCalled();
  });

  it('resets the highlight when the query changes', async () => {
    // Otherwise an index from a longer list can point past the end of a shorter one.
    const { commands } = setup();

    await userEvent.keyboard('{ArrowDown}{ArrowDown}');
    await userEvent.type(screen.getByTestId('palette-input'), 'upload');
    await userEvent.keyboard('{Enter}');

    expect(commands[1].run).toHaveBeenCalled();
  });

  it('never runs a disabled command', async () => {
    const commands = makeCommands([{ disabled: true }]);
    setup(commands);

    await userEvent.keyboard('{Enter}');

    expect(commands[0].run).not.toHaveBeenCalled();
  });

  it('closes on Escape and on a backdrop click', async () => {
    const { onClose } = setup();
    await userEvent.keyboard('{Escape}');
    expect(onClose).toHaveBeenCalledTimes(1);

    await userEvent.click(screen.getByTestId('palette-backdrop'));
    expect(onClose).toHaveBeenCalledTimes(2);
  });

  it('does not close when the click lands inside the dialog', async () => {
    const { onClose } = setup();

    await userEvent.click(screen.getByTestId('palette-input'));

    expect(onClose).not.toHaveBeenCalled();
  });

  it('exposes dialog semantics and the active option', async () => {
    setup();

    const dialog = screen.getByTestId('command-palette');
    expect(dialog).toHaveAttribute('role', 'dialog');
    expect(dialog).toHaveAttribute('aria-modal', 'true');
    // Screen readers follow the highlight through aria-activedescendant, since focus
    // stays in the input while the arrow keys move through the list.
    expect(screen.getByTestId('palette-input')).toHaveAttribute(
      'aria-activedescendant',
      'palette-cmd-new',
    );
    await userEvent.keyboard('{ArrowDown}');
    expect(screen.getByTestId('palette-input')).toHaveAttribute(
      'aria-activedescendant',
      'palette-cmd-upload',
    );
  });
});
