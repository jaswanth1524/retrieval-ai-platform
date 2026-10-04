import { describe, expect, it } from 'vitest';
import { validateUploads } from '../../src/utils/uploadValidation';

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
