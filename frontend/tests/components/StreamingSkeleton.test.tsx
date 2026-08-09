import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import StreamingSkeleton from '../../src/components/StreamingSkeleton';

describe('StreamingSkeleton', () => {
  it('renders the given stage text', () => {
    render(<StreamingSkeleton stage="retrieving…" />);
    expect(screen.getByTestId('streaming-skeleton')).toHaveTextContent('retrieving…');
  });

  it('re-renders with an updated stage', () => {
    const { rerender } = render(<StreamingSkeleton stage="retrieving…" />);
    rerender(<StreamingSkeleton stage="generating answer…" />);
    expect(screen.getByTestId('streaming-skeleton')).toHaveTextContent('generating answer…');
  });
});
