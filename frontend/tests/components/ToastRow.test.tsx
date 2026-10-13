import { render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import ToastRow from '../../src/components/ToastRow';
import type { Toast } from '../../src/hooks/useToasts';

function toast(id: string, tone: Toast['tone'], title: string): Toast {
  return { id, tone, title };
}

describe('ToastRow', () => {
  it('keeps both live regions mounted while empty', () => {
    render(<ToastRow toasts={[]} onDismiss={() => {}} />);
    expect(document.querySelector('[aria-live="assertive"]')).toBeEmptyDOMElement();
    expect(document.querySelector('[aria-live="polite"]')).toBeEmptyDOMElement();
    // No empty alert sitting in the accessibility tree.
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('interrupts for failures and queues confirmations, never one region inside another', () => {
    render(
      <ToastRow
        toasts={[toast('a', 'good', 'Saved'), toast('b', 'bad', 'Upload failed')]}
        onDismiss={() => {}}
      />,
    );
    const assertive = document.querySelector<HTMLElement>('[aria-live="assertive"]')!;
    const polite = document.querySelector<HTMLElement>('[aria-live="polite"]')!;

    expect(within(assertive).getByText('Upload failed')).toBeInTheDocument();
    expect(within(polite).getByText('Saved')).toBeInTheDocument();
    expect(assertive.parentElement?.closest('[aria-live]')).toBeNull();
    expect(polite.querySelector('[aria-live], [role="alert"]')).toBeNull();
  });
});
