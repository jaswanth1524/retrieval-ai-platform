import { useEffect, useRef, useState } from 'react';

function carriesFiles(event: DragEvent): boolean {
  return Array.from(event.dataTransfer?.types ?? []).includes('Files');
}

/** Accept files dropped anywhere on the page.
 *
 * Without a window-level handler, a PDF dropped outside the corpus panel's dropzone
 * made the browser navigate to it, unloading the app and any answer in progress.
 * Drops the dropzone already handled (it calls preventDefault) are left alone, and so
 * are drags that carry no files (text, links). Returns whether files are being
 * dragged over the window, for an overlay. */
export function useWindowFileDrop(enabled: boolean, onFiles: (files: File[]) => void): boolean {
  const [dragging, setDragging] = useState(false);
  // dragenter/dragleave fire for every child crossed; a depth count tells when the
  // drag has really left the window.
  const depth = useRef(0);
  const onFilesRef = useRef(onFiles);
  onFilesRef.current = onFiles;

  useEffect(() => {
    const reset = () => {
      depth.current = 0;
      setDragging(false);
    };
    const handleDragEnter = (event: DragEvent) => {
      if (!carriesFiles(event)) return;
      depth.current += 1;
      if (enabled) setDragging(true);
    };
    const handleDragLeave = (event: DragEvent) => {
      if (!carriesFiles(event)) return;
      depth.current = Math.max(0, depth.current - 1);
      if (depth.current === 0) setDragging(false);
    };
    const handleDragOver = (event: DragEvent) => {
      if (!carriesFiles(event)) return;
      // Always, even when disabled: this is what stops the browser opening the file.
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = enabled ? 'copy' : 'none';
    };
    const handleDrop = (event: DragEvent) => {
      if (!carriesFiles(event)) return;
      reset();
      if (event.defaultPrevented) return; // the corpus dropzone took it
      event.preventDefault();
      const files = Array.from(event.dataTransfer?.files ?? []);
      if (enabled && files.length > 0) onFilesRef.current(files);
    };
    window.addEventListener('dragenter', handleDragEnter);
    window.addEventListener('dragleave', handleDragLeave);
    window.addEventListener('dragover', handleDragOver);
    window.addEventListener('drop', handleDrop);
    return () => {
      window.removeEventListener('dragenter', handleDragEnter);
      window.removeEventListener('dragleave', handleDragLeave);
      window.removeEventListener('dragover', handleDragOver);
      window.removeEventListener('drop', handleDrop);
    };
  }, [enabled]);

  return enabled && dragging;
}
