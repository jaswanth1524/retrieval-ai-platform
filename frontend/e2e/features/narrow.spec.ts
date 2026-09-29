import { expect, test, traceSummary } from '../fixtures';

// Phone width: the context panel and the inspector are drawers, not grid columns.
test.use({ viewport: { width: 390, height: 844 } });

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await expect(page.getByTestId('question-textarea')).toBeEnabled();
});

test('the rail opens the context panel as a drawer and toggles it closed again', async ({ page }) => {
  const panel = page.getByTestId('context-panel');
  await expect(panel).toBeHidden();

  await page.getByTestId('rail-corpus').click();
  const drawer = page.getByRole('dialog', { name: 'Corpus' });
  await expect(drawer).toBeVisible();
  await expect(drawer.getByTestId('corpus-panel-item').filter({ hasText: 'guide.md' })).toBeVisible();

  // Same rail button again closes it; the backdrop and Escape do too.
  await page.getByTestId('rail-corpus').click();
  await expect(drawer).toBeHidden();

  await page.getByTestId('rail-chat').click();
  await expect(page.getByRole('dialog', { name: 'Conversations' })).toBeVisible();
  await page.getByTestId('context-panel-backdrop').click({ position: { x: 380, y: 400 } });
  await expect(page.getByTestId('context-panel-backdrop')).toBeHidden();

  await page.getByTestId('rail-traces').click();
  await expect(page.getByRole('dialog', { name: 'Traces' })).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.getByRole('dialog', { name: 'Traces' })).toBeHidden();
});

test('starting a new conversation from the drawer closes it', async ({ page }) => {
  await page.getByTestId('rail-chat').click();
  const drawer = page.getByRole('dialog', { name: 'Conversations' });
  await drawer.getByTestId('context-panel-action').click();
  await expect(drawer).toBeHidden();
  await expect(page.getByTestId('question-textarea')).toBeVisible();
});

test('the header toggle opens the inspector as a drawer, and Escape returns focus', async ({
  page,
}) => {
  const toggle = page.getByTestId('toggle-inspector');
  await toggle.click();
  const inspector = page.getByRole('dialog', { name: 'Inspector' });
  await expect(inspector).toBeVisible();
  await expect(inspector.getByRole('tab', { name: 'sources' })).toBeFocused();

  await page.keyboard.press('Escape');
  await expect(inspector).toBeHidden();
  await expect(toggle).toBeFocused();
});

test('picking a trace swaps the panel drawer for the inspector drawer', async ({ page, api }) => {
  api.traces = [traceSummary('trace-9', 'How is retrieval fused?')];
  await page.getByTestId('rail-traces').click();
  await page.getByTestId('traces-panel-row').click();

  await expect(page.getByRole('dialog', { name: 'Traces' })).toBeHidden();
  const inspector = page.getByRole('dialog', { name: 'Inspector' });
  await expect(inspector).toBeVisible();
  await expect(inspector.getByRole('tab', { name: 'trace' })).toHaveAttribute('aria-selected', 'true');
});
