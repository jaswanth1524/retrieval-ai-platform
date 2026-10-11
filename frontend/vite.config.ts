import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { brotliCompressSync, constants, gzipSync } from 'node:zlib';
import react from '@vitejs/plugin-react';
import type { Plugin } from 'vite';
import { configDefaults, defineConfig } from 'vitest/config';

// Text assets get .br and .gz copies next to them, which api/static_files.py serves to
// browsers that accept them; the API compresses nothing at request time. Fonts and
// images are already compressed; index.html is tiny and must keep its ETag revalidation.
const PRECOMPRESS = /\.(js|css|svg|json)$/;
const PRECOMPRESS_MIN_BYTES = 1024;

function precompress(): Plugin {
  let assetsDir = '';
  return {
    name: 'docrag-precompress',
    apply: 'build',
    configResolved(config) {
      assetsDir = resolve(config.root, config.build.outDir, config.build.assetsDir);
    },
    closeBundle() {
      for (const name of readdirSync(assetsDir)) {
        if (!PRECOMPRESS.test(name)) continue;
        const path = join(assetsDir, name);
        const content = readFileSync(path);
        if (content.length < PRECOMPRESS_MIN_BYTES) continue;
        writeFileSync(
          `${path}.br`,
          brotliCompressSync(content, {
            params: { [constants.BROTLI_PARAM_QUALITY]: constants.BROTLI_MAX_QUALITY },
          }),
        );
        writeFileSync(`${path}.gz`, gzipSync(content, { level: 9 }));
      }
    },
  };
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), precompress()],
  build: {
    rollupOptions: {
      output: {
        // React changes far less often than app code; its own chunk keeps a returning
        // browser's cached copy valid across app-only deploys. Everything else is left
        // to Rollup, which puts the markdown renderer's packages in the lazy Markdown
        // chunk (only that chunk imports them) — a catch-all "vendor" chunk would be
        // loaded up front the moment any eagerly imported package landed in it.
        manualChunks(id) {
          if (/[\\/]node_modules[\\/](react|react-dom|scheduler)[\\/]/.test(id)) return 'react';
          return undefined;
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/health': 'http://localhost:8000',
      '/config': 'http://localhost:8000',
      '/documents': 'http://localhost:8000',
      '/questions': 'http://localhost:8000',
      '/traces': 'http://localhost:8000',
      '/feedback': 'http://localhost:8000',
      // Without these the dev server answered with index.html: "Download a backup"
      // saved the app's HTML as the zip.
      '/export': 'http://localhost:8000',
      '/metrics': 'http://localhost:8000',
      '/access': 'http://localhost:8000',
      '/import': 'http://localhost:8000',
      '/search': 'http://localhost:8000',
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./tests/setup.ts'],
    globals: true,
    // Playwright specs (npm run e2e) — a different runner.
    exclude: [...configDefaults.exclude, 'e2e/**'],
  },
});
