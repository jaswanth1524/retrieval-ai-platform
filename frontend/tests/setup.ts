import '@testing-library/jest-dom/vitest';

// jsdom does not implement scrollIntoView; ChatThread calls it on every turn/pending
// change, which would otherwise throw in any test that renders it.
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {};
}
