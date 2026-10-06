import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

// The theme's small-text colours must stay readable: WCAG AA asks 4.5:1 for body text.
const css = readFileSync(resolve(__dirname, '../../src/styles/tokens.css'), 'utf8');

function block(selector: string): string {
  const start = css.indexOf(selector);
  return css.slice(start, css.indexOf('}', start));
}

function token(theme: string, name: string): string {
  const match = block(theme).match(new RegExp(`--${name}:\\s*([^;]+);`));
  if (!match) throw new Error(`--${name} not found`);
  return match[1].trim();
}

function srgb(value: string): number[] {
  if (value.startsWith('#')) {
    return [1, 3, 5].map((index) => parseInt(value.slice(index, index + 2), 16) / 255);
  }
  const [l, c, h] = value.match(/oklch\(([\d.]+)%\s+([\d.]+)\s+([\d.]+)\)/)!.slice(1).map(Number);
  const [L, a, b] = [l / 100, c * Math.cos((h * Math.PI) / 180), c * Math.sin((h * Math.PI) / 180)];
  const l3 = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m3 = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s3 = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  const linear = [
    4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3,
    -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3,
    -0.0041960863 * l3 - 0.7034186147 * m3 + 1.707614701 * s3,
  ];
  return linear.map((x) => Math.min(1, Math.max(0, x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055)));
}

function luminance(value: string): number {
  const [r, g, b] = srgb(value).map((c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

describe('theme tokens', () => {
  for (const theme of [":root[data-theme='dark']", ":root[data-theme='light']"]) {
    for (const text of ['dim', 'muted', 'warn']) {
      it(`--${text} is readable on every surface in ${theme}`, () => {
        for (const surface of ['paper', 'card', 'sunk']) {
          expect(contrast(token(theme, text), token(theme, surface))).toBeGreaterThanOrEqual(4.5);
        }
      });
    }
  }
});
