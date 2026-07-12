import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import DocumentFilter from '../../src/components/DocumentFilter';

describe('DocumentFilter', () => {
  it('renders nothing when there are no indexed documents', () => {
    const { container } = render(
      <DocumentFilter filenames={[]} selected={[]} onChange={vi.fn()} />,
    );

    expect(container).toBeEmptyDOMElement();
  });

  it('checks "All documents" when nothing is selected', () => {
    render(<DocumentFilter filenames={['a.txt', 'b.txt']} selected={[]} onChange={vi.fn()} />);

    expect(screen.getByTestId('document-filter-all')).toBeChecked();
  });

  it('selecting a document unchecks "All documents" and checks that document', () => {
    render(
      <DocumentFilter filenames={['a.txt', 'b.txt']} selected={['a.txt']} onChange={vi.fn()} />,
    );

    expect(screen.getByTestId('document-filter-all')).not.toBeChecked();
    const items = screen.getAllByTestId('document-filter-item');
    expect(items[0].querySelector('input')).toBeChecked();
    expect(items[1].querySelector('input')).not.toBeChecked();
  });

  it('clicking a document filename toggles it into the selection', async () => {
    const onChange = vi.fn();
    render(
      <DocumentFilter filenames={['a.txt', 'b.txt']} selected={[]} onChange={onChange} />,
    );

    const items = screen.getAllByTestId('document-filter-item');
    await userEvent.click(items[0].querySelector('input')!);

    expect(onChange).toHaveBeenCalledWith(['a.txt']);
  });

  it('clicking an already-selected filename removes it', async () => {
    const onChange = vi.fn();
    render(
      <DocumentFilter filenames={['a.txt', 'b.txt']} selected={['a.txt', 'b.txt']} onChange={onChange} />,
    );

    const items = screen.getAllByTestId('document-filter-item');
    await userEvent.click(items[0].querySelector('input')!);

    expect(onChange).toHaveBeenCalledWith(['b.txt']);
  });

  it('clicking "All documents" clears the selection', async () => {
    const onChange = vi.fn();
    render(
      <DocumentFilter filenames={['a.txt']} selected={['a.txt']} onChange={onChange} />,
    );

    await userEvent.click(screen.getByTestId('document-filter-all'));

    expect(onChange).toHaveBeenCalledWith([]);
  });

  it('disables all checkboxes when disabled is true', () => {
    render(
      <DocumentFilter filenames={['a.txt']} selected={[]} onChange={vi.fn()} disabled />,
    );

    expect(screen.getByTestId('document-filter-all')).toBeDisabled();
    expect(screen.getByTestId('document-filter-item').querySelector('input')).toBeDisabled();
  });
});
