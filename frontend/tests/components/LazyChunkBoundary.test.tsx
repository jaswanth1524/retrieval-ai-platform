import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { describe, expect, it, vi } from 'vitest';
import LazyChunkBoundary from '../../src/components/LazyChunkBoundary';

function Broken(): never {
  throw new Error('Failed to fetch dynamically imported module');
}

describe('LazyChunkBoundary', () => {
  it('replaces a panel whose chunk failed with a reload prompt, and dismisses cleanly', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.spyOn(console, 'warn').mockImplementation(() => {});

    function Harness() {
      const [open, setOpen] = useState(true);
      return (
        <>
          <p>chat still here</p>
          <LazyChunkBoundary onDismiss={() => setOpen(false)}>{open && <Broken />}</LazyChunkBoundary>
        </>
      );
    }
    render(<Harness />);

    expect(screen.getByRole('alert')).toHaveTextContent('DocRAG was updated');
    expect(screen.getByText('chat still here')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Dismiss' }));
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    vi.restoreAllMocks();
  });
});
