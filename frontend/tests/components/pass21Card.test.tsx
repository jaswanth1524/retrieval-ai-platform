import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import DocumentCard from '../../src/components/corpus/DocumentCard';
import type { DocCard } from '../../src/components/corpus/types';

function card(sourceUrl: string): DocCard {
  return {
    key: 'doc:policy.html',
    filename: 'policy.html',
    status: 'indexed',
    detail: '3 chunks',
    pct: 100,
    tags: [],
    sourceUrl,
  };
}

function show(sourceUrl: string) {
  render(
    <DocumentCard
      card={card(sourceUrl)}
      readOnly={false}
      confirming={false}
      deleting={false}
      selectable={false}
      selected={false}
      onToggleSelect={vi.fn()}
      onAskDelete={vi.fn()}
      onCancelDelete={vi.fn()}
      onConfirmDelete={vi.fn()}
    />,
  );
  return screen.getByTestId('corpus-source-link');
}

describe('pass 21: a document’s source address', () => {
  it('is a link only when it is a web address', () => {
    expect(show('javascript:alert(1)')).not.toHaveAttribute('href');
  });

  it('links a web address', () => {
    expect(show('https://example.com/policy.html')).toHaveAttribute('href', 'https://example.com/policy.html');
  });
});
