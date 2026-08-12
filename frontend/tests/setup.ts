import '@testing-library/jest-dom/vitest';

// jsdom does not implement scrollIntoView; DocumentViewer calls it to bring a cited
// chunk into view, which would otherwise throw in any test that renders it. (ChatThread
// used to call it too, but now drives its own scroll container's scrollTop directly.)
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {};
}

// jsdom implements no media queries at all, and useResponsiveLayout calls matchMedia
// during App's first render. Default `matches: false` means neither breakpoint is
// active, so the inspector keeps its closed default — a test that cares about a
// breakpoint stubs window.matchMedia itself.
if (!window.matchMedia) {
  window.matchMedia = (query: string): MediaQueryList =>
    ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
      dispatchEvent: () => false,
    }) as MediaQueryList;
}
