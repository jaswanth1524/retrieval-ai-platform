import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import CollectedPanel from '../../src/components/CollectedPanel';
import {
  MAX_BASKET,
  addToBasket,
  quotesToCsv,
  quotesToMarkdown,
  readBasket,
  readQuotes,
  removeQuote,
  saveQuote,
  type BasketPassage,
  type Quote,
} from '../../src/utils/collected';

afterEach(() => localStorage.clear());

const here = { origin: 'https://docrag.example', pathname: '/' };

function quote(id: string, text: string, extra: Partial<Quote> = {}): Quote {
  return {
    id,
    text,
    source: 'guide.pdf, p. 3, Setup',
    filename: 'guide.pdf',
    chunkId: 'c1',
    workspace: '',
    savedAt: Number(id),
    ...extra,
  };
}

function passage(chunkId: string, extra: Partial<BasketPassage> = {}): BasketPassage {
  return {
    chunkId,
    filename: 'guide.pdf',
    page: 3,
    section: 'Setup',
    snippet: `text of ${chunkId}`,
    workspace: '',
    addedAt: 1,
    ...extra,
  };
}

describe('quotes and the passage basket', () => {
  it('keeps quotes newest first, once each, and exports them with passage links', () => {
    saveQuote(quote('1', 'Install first.'));
    saveQuote(quote('2', 'Then run it.', { workspace: 'legal' }));
    saveQuote(quote('3', 'Install first.'));
    expect(readQuotes().map((q) => q.text)).toEqual(['Install first.', 'Then run it.']);
    const markdown = quotesToMarkdown(readQuotes(), here);
    expect(markdown).toContain('> Install first.\n\n— [guide.pdf, p. 3, Setup](https://docrag.example/#d=guide.pdf&k=c1)');
    expect(markdown).toContain('w=legal');
    expect(quotesToCsv(readQuotes(), here)).toMatch(/^quote,source,link,saved\r?\n/);
    expect(removeQuote('3').map((q) => q.id)).toEqual(['2']);
  });

  it('ignores damaged storage and keeps the basket in picking order, capped', () => {
    localStorage.setItem('docrag-quotes', '{not json');
    expect(readQuotes()).toEqual([]);
    addToBasket([passage('a'), passage('b')]);
    addToBasket([passage('a'), passage('c')]);
    expect(readBasket().map((p) => p.chunkId)).toEqual(['a', 'b', 'c']);
    addToBasket(Array.from({ length: MAX_BASKET }, (_, i) => passage(`x${i}`)));
    expect(readBasket()).toHaveLength(MAX_BASKET);
    expect(readBasket()[0].chunkId).toBe('x0');
  });

  it('asks about the basket with a question and leaves out another workspace’s passages', async () => {
    const onAskAbout = vi.fn();
    const onScopeTo = vi.fn();
    render(
      <CollectedPanel
        quotes={[quote('1', 'Install first.')]}
        basket={[passage('a'), passage('b', { filename: 'other.pdf' }), passage('z', { workspace: 'legal' })]}
        workspace=""
        onOpenPassage={vi.fn()}
        onRemoveQuote={vi.fn()}
        onRemovePassages={vi.fn()}
        onClearBasket={vi.fn()}
        onAskAbout={onAskAbout}
        onScopeTo={onScopeTo}
      />,
    );
    expect(screen.getByTestId('basket-elsewhere')).toHaveTextContent('1 passage is from another workspace');
    expect(screen.getByTestId('basket-ask')).toBeDisabled();
    await userEvent.type(screen.getByTestId('basket-question'), 'What changed?');
    await userEvent.click(screen.getByTestId('basket-ask'));
    expect(onAskAbout).toHaveBeenCalledWith('What changed?', ['a', 'b']);
    await userEvent.click(screen.getByTestId('basket-scope'));
    expect(onScopeTo).toHaveBeenCalledWith(['guide.pdf', 'other.pdf']);
    expect(screen.getAllByTestId('quote-row')).toHaveLength(1);
  });
});
