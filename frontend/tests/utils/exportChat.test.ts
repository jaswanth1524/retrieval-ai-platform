import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { ChatTurn } from '../../src/components/ChatMessage';
import { chatToJson, chatToMarkdown, downloadFile } from '../../src/utils/exportChat';

function makeTurn(overrides: Partial<ChatTurn> = {}): ChatTurn {
  return {
    id: 't1',
    role: 'assistant',
    content: 'The answer is 42.',
    sources: [],
    timestamp: 0,
    timings: null,
    traceId: null,
    ...overrides,
  };
}

describe('chatToMarkdown', () => {
  it('renders speaker, content, and sources per turn', () => {
    const md = chatToMarkdown([
      makeTurn({ role: 'user', content: 'What is the answer?' }),
      makeTurn({
        role: 'assistant',
        content: 'The answer is 42.',
        sources: [{ source_number: 1, filename: 'doc.pdf', page: 3, section: 'Intro', chunk_id: 'c1', text: '' }],
      }),
    ]);
    expect(md).toContain('You');
    expect(md).toContain('What is the answer?');
    expect(md).toContain('DocRAG');
    expect(md).toContain('The answer is 42.');
    expect(md).toContain('[1] doc.pdf · p.3 · Intro');
  });
});

describe('chatToJson', () => {
  it('round-trips turns as JSON', () => {
    const turns = [makeTurn()];
    const json = chatToJson(turns);
    expect(JSON.parse(json)).toEqual(turns);
  });
});

function readBlobAsText(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsText(blob);
  });
}

describe('downloadFile', () => {
  let createObjectURL: ReturnType<typeof vi.fn>;
  let revokeObjectURL: ReturnType<typeof vi.fn>;
  let capturedBlob: Blob | undefined;

  beforeEach(() => {
    capturedBlob = undefined;
    createObjectURL = vi.fn((blob: Blob) => {
      capturedBlob = blob;
      return 'blob:mock-url';
    });
    revokeObjectURL = vi.fn();
    // jsdom has no createObjectURL/revokeObjectURL implementation.
    URL.createObjectURL = createObjectURL as unknown as typeof URL.createObjectURL;
    URL.revokeObjectURL = revokeObjectURL as unknown as typeof URL.revokeObjectURL;
  });

  it('creates a Blob whose type is the mimeType and whose body is the content', async () => {
    downloadFile('chat.md', 'text/markdown', '## You\n\nHello');

    expect(createObjectURL).toHaveBeenCalledTimes(1);
    expect(capturedBlob).toBeDefined();
    expect(capturedBlob!.type).toBe('text/markdown');

    const body = await readBlobAsText(capturedBlob!);
    expect(body).toBe('## You\n\nHello');
    // Regression guard: a swapped-argument call site would make the body equal the
    // literal mimeType string instead of real content.
    expect(body).not.toBe('text/markdown');
  });

  it('revokes the object URL after triggering the download', () => {
    downloadFile('chat.json', 'application/json', '{}');
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:mock-url');
  });
});
