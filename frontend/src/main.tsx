import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './styles/tokens.css';
import './styles/global.css';
import App from './App';

// Applied before the first render so CSS variables resolve to the stored theme
// immediately — avoids a flash of the default (dark) theme for light-mode users.
// localStorage can throw in some private-browsing/storage-denied contexts; falling
// back to the default theme there is preferable to the app failing to boot at all.
let storedTheme: string | null = null;
try {
  storedTheme = localStorage.getItem('docrag-theme');
} catch {
  storedTheme = null;
}
// No saved choice yet: follow the operating system's setting instead of always dark.
const prefersLight =
  storedTheme === null && window.matchMedia?.('(prefers-color-scheme: light)').matches;
document.documentElement.dataset.theme =
  storedTheme === 'light' || (storedTheme === null && prefersLight) ? 'light' : 'dark';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
