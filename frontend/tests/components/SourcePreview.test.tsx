import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import SourcePreview from '../../src/components/SourcePreview';

describe('SourcePreview', () => {
  it('renders the citation index, filename, location, and quote', () => {
    render(
      <SourcePreview
        citation={{
          source_number: 2,
          filename: 'acme-msa-2024.pdf',
          page: 19,
          section: '§11.3',
          chunk_id: 'chunk-2',
          text: 'The limitations in Section 11.2 shall not apply to breaches of Section 9.',
        }}
      />,
    );

    const preview = screen.getByTestId('source-preview');
    expect(preview).toHaveTextContent('2');
    expect(preview).toHaveTextContent('acme-msa-2024.pdf');
    expect(preview).toHaveTextContent('p.19');
    expect(preview).toHaveTextContent('§11.3');
    expect(preview).toHaveTextContent('The limitations in Section 11.2 shall not apply to breaches of Section 9.');
  });

  it('shows the passage the answer rests on instead of the chunk opening when it has one', () => {
    render(
      <SourcePreview
        citation={{
          source_number: 1,
          filename: 'policy.md',
          page: 1,
          section: 'Leave',
          chunk_id: 'c1',
          text: 'The policy applies to all staff.',
          spans: [
            { start: 33, end: 70, quote: 'Employees accrue twenty days a year.' },
            { start: 71, end: 99, quote: 'Unused days expire in March.' },
          ],
        }}
      />,
    );

    expect(screen.getByTestId('source-preview')).toHaveTextContent(
      'Employees accrue twenty days a year. … Unused days expire in March.',
    );
    expect(screen.getByTestId('source-preview')).not.toHaveTextContent('applies to all staff');
  });
});

