import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import CitationCard from '../../src/components/CitationCard';

describe('CitationCard', () => {
  it('renders all fields including the excerpt text', () => {
    render(
      <CitationCard
        sourceNumber={1}
        filename="guide.md"
        page={3}
        section="Setup"
        chunkId="8835c8aed15c331e74e8"
        text="Start Docker Compose which launches Qdrant, the API, and the UI."
      />,
    );

    expect(screen.getByTestId('citation-card')).toBeInTheDocument();
    expect(screen.getByText('[1]')).toBeInTheDocument();
    expect(screen.getByText('guide.md')).toBeInTheDocument();
    expect(screen.getByText(/p\.3/)).toBeInTheDocument();
    expect(screen.getByText(/Setup/)).toBeInTheDocument();
    expect(screen.getByText(/8835c8aed15c331e74e8/)).toBeInTheDocument();
    expect(
      screen.getByText('Start Docker Compose which launches Qdrant, the API, and the UI.'),
    ).toBeInTheDocument();
  });
});
