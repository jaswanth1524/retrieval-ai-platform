import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../src/App', () => ({
  default: () => null,
}));

async function importMain(): Promise<void> {
  vi.resetModules();
  await import('../src/main');
}

describe('main.tsx theme init', () => {
  beforeEach(() => {
    document.body.innerHTML = '<div id="root"></div>';
    delete document.documentElement.dataset.theme;
  });

  afterEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  it('applies the stored light theme before first render', async () => {
    localStorage.setItem('docrag-theme', 'light');

    await importMain();

    expect(document.documentElement.dataset.theme).toBe('light');
  });

  it('defaults to dark when nothing is stored', async () => {
    await importMain();

    expect(document.documentElement.dataset.theme).toBe('dark');
  });

  it('defaults to dark when localStorage.getItem throws', async () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new DOMException('denied', 'SecurityError');
    });

    await expect(importMain()).resolves.toBeUndefined();
    expect(document.documentElement.dataset.theme).toBe('dark');
  });
});
