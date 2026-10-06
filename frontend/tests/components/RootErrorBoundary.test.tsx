import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import RootErrorBoundary from '../../src/components/RootErrorBoundary';
import { CHAT_STORAGE_KEY } from '../../src/hooks/chatPersistence';

function Broken(): never {
  throw new Error('bad turn');
}

describe('RootErrorBoundary', () => {
  afterEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  it('offers to save and then clear the history instead of a blank page', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    localStorage.setItem(CHAT_STORAGE_KEY, '{"version":2,"conversations":[]}');
    const reload = vi.fn();
    vi.spyOn(window, 'location', 'get').mockReturnValue({ ...window.location, reload });
    const createObjectURL = vi.fn(() => 'blob:x');
    vi.stubGlobal('URL', { ...URL, createObjectURL, revokeObjectURL: vi.fn() });

    render(
      <RootErrorBoundary>
        <Broken />
      </RootErrorBoundary>,
    );

    expect(screen.getByTestId('root-error')).toBeInTheDocument();
    await userEvent.click(screen.getByTestId('root-error-export'));
    expect(createObjectURL).toHaveBeenCalledTimes(1);
    await userEvent.click(screen.getByTestId('root-error-clear'));
    expect(localStorage.getItem(CHAT_STORAGE_KEY)).toBeNull();
    expect(reload).toHaveBeenCalled();
    vi.unstubAllGlobals();
  });
});
