import type { ReactNode } from 'react';

// A link followed in this tab used to unload the app mid-answer. `noreferrer` also
// keeps the DocRAG URL out of wherever a document's link points.
export default function ExternalLink({ href, children }: { href?: string; children?: ReactNode }) {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  );
}
