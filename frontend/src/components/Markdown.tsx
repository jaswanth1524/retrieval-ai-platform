import { memo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { splitStreamingMarkdown } from '../utils/markdownBlocks';
import type { MarkdownProps } from './PlainAnswer';

const REMARK_PLUGINS = [remarkGfm];

const MarkdownBlock = memo(function MarkdownBlock({ text, components }: MarkdownProps) {
  return (
    <ReactMarkdown remarkPlugins={REMARK_PLUGINS} components={components}>
      {text}
    </ReactMarkdown>
  );
});

/** An answer rendered as markdown. While it streams, finished blocks are memoized, so
 *  each token re-parses only the block it extends rather than the whole answer so far.
 *
 *  Its own chunk (see markdownLoader): the parser is most of the app's third-party code
 *  and nothing needs it before the first answer. */
function Markdown({ text, components, streaming = false }: MarkdownProps) {
  if (!streaming) return <MarkdownBlock text={text} components={components} />;
  const { done, tail } = splitStreamingMarkdown(text);
  return (
    <>
      {done.map((block, index) => (
        <MarkdownBlock key={index} text={block} components={components} />
      ))}
      <MarkdownBlock text={tail} components={components} />
    </>
  );
}

export default Markdown;
