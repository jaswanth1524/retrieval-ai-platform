import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import CorpusPanel from '../../src/components/CorpusPanel';

describe('CorpusPanel', () => {
  it('renders nothing when there are no indexed documents', () => {
    const { container } = render(<CorpusPanel filenames={[]} onDelete={vi.fn()} />);

    expect(container).toBeEmptyDOMElement();
  });

  it('lists indexed filenames', () => {
    render(<CorpusPanel filenames={['a.txt', 'b.pdf']} onDelete={vi.fn()} />);

    expect(screen.getByText('a.txt')).toBeInTheDocument();
    expect(screen.getByText('b.pdf')).toBeInTheDocument();
  });

  it('clicking delete shows an inline confirm, and confirming calls onDelete once', async () => {
    const onDelete = vi.fn().mockResolvedValue(undefined);
    render(<CorpusPanel filenames={['a.txt']} onDelete={onDelete} />);

    await userEvent.click(screen.getByLabelText('Delete a.txt'));
    expect(screen.getByText('Delete?')).toBeInTheDocument();

    await userEvent.click(screen.getByText('Confirm'));

    expect(onDelete).toHaveBeenCalledTimes(1);
    expect(onDelete).toHaveBeenCalledWith('a.txt');
  });

  it('clicking cancel dismisses the confirm without calling onDelete', async () => {
    const onDelete = vi.fn();
    render(<CorpusPanel filenames={['a.txt']} onDelete={onDelete} />);

    await userEvent.click(screen.getByLabelText('Delete a.txt'));
    await userEvent.click(screen.getByText('Cancel'));

    expect(onDelete).not.toHaveBeenCalled();
    expect(screen.queryByText('Delete?')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Delete a.txt')).toBeInTheDocument();
  });

  it('shows an error message when onDelete rejects', async () => {
    const onDelete = vi.fn().mockRejectedValue(new Error('No indexed document named a.txt.'));
    render(<CorpusPanel filenames={['a.txt']} onDelete={onDelete} />);

    await userEvent.click(screen.getByLabelText('Delete a.txt'));
    await userEvent.click(screen.getByText('Confirm'));

    expect(await screen.findByRole('alert')).toHaveTextContent('No indexed document named a.txt.');
  });

  it('disables delete buttons when disabled is true', () => {
    render(<CorpusPanel filenames={['a.txt']} onDelete={vi.fn()} disabled />);

    expect(screen.getByLabelText('Delete a.txt')).toBeDisabled();
  });
});
