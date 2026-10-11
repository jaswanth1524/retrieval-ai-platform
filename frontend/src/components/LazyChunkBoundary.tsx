import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';
import './LazyChunkBoundary.css';

interface LazyChunkBoundaryProps {
  children: ReactNode;
  /** Close whatever failed to open, so dismissing doesn't just retry the same failure. */
  onDismiss: () => void;
}

interface LazyChunkBoundaryState {
  failed: boolean;
  /** The failure was a code chunk that couldn't be fetched, not a render error. */
  chunkMissing: boolean;
}

// What each engine says when a dynamic import's file is gone (Chrome, Firefox, Safari,
// and Vite's CSS preload).
const CHUNK_LOAD_ERROR =
  /dynamically imported module|Importing a module script failed|Unable to preload|ChunkLoadError/i;

function isChunkLoadError(error: unknown): boolean {
  return error instanceof Error && CHUNK_LOAD_ERROR.test(`${error.name} ${error.message}`);
}

/** Catches a lazy-loaded dialog whose code chunk can't be fetched.
 *
 * A tab left open across a redeploy still references the previous build's hashed chunk
 * names; opening Settings or the palette then requests a file the new server doesn't
 * have. Uncaught, that rejection unmounts the whole app to a blank page — here it only
 * costs the dialog, with a reload offered to pick up the new build.
 */
class LazyChunkBoundary extends Component<LazyChunkBoundaryProps, LazyChunkBoundaryState> {
  state: LazyChunkBoundaryState = { failed: false, chunkMissing: false };

  // Any render error in the panel lands here too: blaming those on a deploy ("DocRAG
  // was updated") sent the user to reload for a fault a reload doesn't fix.
  static getDerivedStateFromError(error: unknown): LazyChunkBoundaryState {
    return { failed: true, chunkMissing: isChunkLoadError(error) };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.warn('A lazy-loaded panel failed to load.', error, info.componentStack);
  }

  private dismiss = () => {
    this.props.onDismiss();
    this.setState({ failed: false, chunkMissing: false });
  };

  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <div className="lazy-chunk-boundary" role="alert" data-testid="lazy-chunk-error">
        <span>
          {this.state.chunkMissing
            ? 'DocRAG was updated. Reload to open this.'
            : "This panel couldn't be shown."}
        </span>
        <button type="button" onClick={() => window.location.reload()}>
          Reload
        </button>
        <button type="button" onClick={this.dismiss} aria-label="Dismiss">
          ✕
        </button>
      </div>
    );
  }
}

export default LazyChunkBoundary;
