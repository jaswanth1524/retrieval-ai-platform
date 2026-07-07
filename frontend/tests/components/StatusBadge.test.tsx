import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import StatusBadge from '../../src/components/StatusBadge';

describe('StatusBadge', () => {
  it('renders the checking label', () => {
    render(<StatusBadge status="checking" />);

    expect(screen.getByText('Checking...')).toBeInTheDocument();
  });

  it('renders the ok label', () => {
    render(<StatusBadge status="ok" />);

    expect(screen.getByText('API online')).toBeInTheDocument();
  });

  it('renders the error label and message', () => {
    render(<StatusBadge status="error" message="Connection refused" />);

    expect(screen.getByText('API unreachable')).toBeInTheDocument();
    expect(screen.getByText('Connection refused')).toBeInTheDocument();
  });

  it('announces status changes to assistive tech via role="status"', () => {
    render(<StatusBadge status="ok" />);

    expect(screen.getByRole('status')).toHaveTextContent('API online');
  });
});
