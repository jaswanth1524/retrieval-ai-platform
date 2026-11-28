import type { Components } from 'react-markdown';
import ExternalLink from './ExternalLink';

/** How an answer's markdown renders: images never load (only their alt text shows, so
 *  a poisoned document can't make the browser fetch a URL), links open in a new tab,
 *  headings are kept small. Shared by the chat and the shared-answer view. */
export const MARKDOWN_COMPONENTS: Components = {
  img: ({ alt }) => (alt ? <span className="chat-message__image-alt">[{alt}]</span> : null),
  a: ({ href, children }) => <ExternalLink href={href}>{children}</ExternalLink>,
  h1: ({ children }) => <h3>{children}</h3>,
  h2: ({ children }) => <h4>{children}</h4>,
  h3: ({ children }) => <h5>{children}</h5>,
  h4: ({ children }) => <h6>{children}</h6>,
  h5: ({ children }) => <h6>{children}</h6>,
  h6: ({ children }) => <h6>{children}</h6>,
  // Wide tables scroll inside the answer instead of widening the page on a phone.
  table: ({ children }) => (
    <div className="chat-message__table-scroll">
      <table>{children}</table>
    </div>
  ),
};
