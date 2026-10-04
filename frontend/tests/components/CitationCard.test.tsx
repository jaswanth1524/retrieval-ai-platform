import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import CitationCard from '../../src/components/CitationCard';

describe('CitationCard', () => {
  it('renders the source number, filename, and location as an inline chip', () => {
    render(<CitationCard sourceNumber={1} filename="guide.md" page={3} section="Setup" />);

    const chip = screen.getByTestId('citation-card');
    expect(chip).toHaveTextContent('1');
    expect(chip).toHaveTextContent('guide.md');
    expect(chip).toHaveTextContent('p.3');
    expect(chip).toHaveTextContent('Setup');
    expect(chip).toHaveAccessibleName('Open guide.md at source 1');
  });

  it('fires onOpen when clicked', async () => {
    const onOpen = vi.fn();
    render(<CitationCard sourceNumber={1} filename="guide.md" page={3} section="Setup" onOpen={onOpen} />);

    await userEvent.click(screen.getByTestId('citation-card'));
    expect(onOpen).toHaveBeenCalledOnce();
  });

  it('fires onHoverStart/onHoverEnd on mouse enter/leave', async () => {
    const onHoverStart = vi.fn();
    const onHoverEnd = vi.fn();
    render(
      <CitationCard
        sourceNumber={1}
        filename="guide.md"
        page={3}
        section="Setup"
        onHoverStart={onHoverStart}
        onHoverEnd={onHoverEnd}
      />,
    );

    await userEvent.hover(screen.getByTestId('citation-card'));
    expect(onHoverStart).toHaveBeenCalledOnce();

    await userEvent.unhover(screen.getByTestId('citation-card'));
    expect(onHoverEnd).toHaveBeenCalledOnce();
  });

  it('does not reopen the preview when focus comes back after opening the source', () => {
    const onHoverStart = vi.fn();
    render(
      <CitationCard sourceNumber={1} filename="guide.md" page={3} section="Setup" onOpen={vi.fn()} onHoverStart={onHoverStart} />,
    );
    const chip = screen.getByTestId('citation-card');

    fireEvent.click(chip);
    fireEvent.focus(chip); // the viewer closed and handed focus back
    expect(onHoverStart).not.toHaveBeenCalled();

    fireEvent.blur(chip);
    fireEvent.focus(chip); // a later keyboard visit still previews
    expect(onHoverStart).toHaveBeenCalledOnce();
  });

  it('does not open the hover preview from a tap', () => {
    const onHoverStart = vi.fn();
    render(<CitationCard sourceNumber={1} filename="guide.md" page={3} section="Setup" onHoverStart={onHoverStart} />);
    const chip = screen.getByTestId('citation-card');

    fireEvent.touchStart(chip);
    fireEvent.mouseEnter(chip);

    expect(onHoverStart).not.toHaveBeenCalled();
  });
});

