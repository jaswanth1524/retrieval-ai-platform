import { describe, expect, it } from 'vitest';
import {
  ACCEPTED_EXTENSIONS,
  acceptedExtensions,
  extensionsLabel,
  formatBytes,
  validateUploads,
} from '../../src/utils/uploadValidation';

describe('validateUploads', () => {
  it('accepts supported types under the size limit and explains every rejection', () => {
    const ok = new File(['x'], 'Guide.PDF');
    const noExtension = new File(['x'], 'README');
    const wrongType = new File(['x'], 'photo.png');
    const tooBig = new File(['x'.repeat(20)], 'big.txt');

    const { accepted, rejected } = validateUploads([ok, noExtension, wrongType, tooBig], 10);

    expect(accepted).toEqual([ok]);
    expect(rejected.map((item) => item.filename)).toEqual(['README', 'photo.png', 'big.txt']);
    expect(rejected[2].message).toContain('too large');
  });
});

describe('formatBytes', () => {
  it('uses the unit that reads naturally', () => {
    // A 12 KB note was shown as "0.0 MB".
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(12 * 1024)).toBe('12 KB');
    expect(formatBytes(50 * 1024 * 1024)).toBe('50.0 MB');
    expect(formatBytes(3 * 1024 ** 3)).toBe('3.0 GB');
  });
});

describe('server-supplied upload types', () => {
  it('accepts what the server lists, falling back to the built-in list', () => {
    expect(acceptedExtensions(['.pdf', '.epub'])).toEqual(['.pdf', '.epub']);
    expect(acceptedExtensions([])).toEqual(ACCEPTED_EXTENSIONS);
    expect(acceptedExtensions(undefined)).toContain('.eml');

    const epub = new File(['x'], 'novel.EPUB');
    const docx = new File(['x'], 'memo.docx');
    const { accepted, rejected } = validateUploads([epub, docx], 1024, ['.pdf', '.epub']);
    expect(accepted).toEqual([epub]);
    expect(rejected[0].message).toContain("memo.docx isn't a supported type (.pdf .epub)");
  });

  it('labels the dropzone without the alternate spellings', () => {
    expect(extensionsLabel(['.pdf', '.md', '.markdown', '.html', '.htm', '.eml'])).toBe(
      'pdf md html eml',
    );
  });
});
