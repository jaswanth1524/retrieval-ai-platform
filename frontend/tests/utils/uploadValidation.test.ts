import { describe, expect, it } from 'vitest';
import { formatBytes, validateUploads } from '../../src/utils/uploadValidation';

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
