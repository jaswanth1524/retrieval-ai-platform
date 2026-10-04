import react from '@vitejs/plugin-react';
import { configDefaults, defineConfig } from 'vitest/config';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        // Third-party code changes far less often than app code; separate chunks keep
        // a returning browser's cached copy valid across app-only deploys.
        manualChunks(id) {
          if (!id.includes('node_modules')) return undefined;
          if (/[\\/]node_modules[\\/](react|react-dom|scheduler)[\\/]/.test(id)) return 'react';
          return 'vendor';
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
