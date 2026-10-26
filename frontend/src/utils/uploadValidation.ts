// Source code and config files (api/parsers/structured.py CODE_EXTENSIONS).
export const CODE_EXTENSIONS = [
  '.py',
  '.js',
  '.ts',
  '.go',
  '.java',
  '.rs',
  '.rb',
  '.c',
  '.cpp',
  '.h',
  '.sh',
  '.sql',
  '.yaml',
  '.yml',
  '.toml',
];

// api/documents.py CORE_EXTENSIONS (every type but images, which need the server's OCR
// extra), for a server too old to report its own (PublicConfigResponse.
// supported_extensions), and until /config has loaded.
export const ACCEPTED_EXTENSIONS = [
  '.pdf',
  '.txt',
  '.md',
  '.markdown',
  '.docx',
  '.html',
  '.htm',
  '.csv',
  '.epub',
  '.eml',
  '.xlsx',
  '.pptx',
  '.odt',
  '.json',
  ...CODE_EXTENSIONS,
];

/** The suffixes to accept: the server's list when it sent one. */
export function acceptedExtensions(fromServer?: readonly string[]): readonly string[] {
  return fromServer && fromServer.length > 0 ? fromServer : ACCEPTED_EXTENSIONS;
}

/** The dropzone's short list ("pdf docx md …"), without the alternate spellings, and
 *  source-code and config types said as one "code". */
export function extensionsLabel(extensions: readonly string[]): string {
  const alternates = new Set(['.markdown', '.htm', '.jpeg', '.tif']);
  const code = new Set(CODE_EXTENSIONS);
  const shown = extensions
    .filter((extension) => !alternates.has(extension) && !code.has(extension))
    .map((extension) => extension.replace(/^\./, ''));
  if (extensions.some((extension) => code.has(extension))) shown.push('code');
  return shown.join(' ');
}

// Matches the backend's AppSettings.max_upload_bytes default — used only until
// /config has loaded and supplies the server's actual configured limit.
export const DEFAULT_MAX_UPLOAD_BYTES = 50 * 1024 * 1024;

export interface RejectedFile {
  filename: string;
  message: string;
}

/** A size in the unit that reads naturally: a 12 KB note showed as "0.0 MB". */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

/** Split files into ones the server will take and ones it would refuse.
 *
 * A file input's `accept` filters the picker only; a drop bypasses it, and an
 * unsupported file used to cost a round trip to learn the server rejects it. */
export function validateUploads(
  files: Iterable<File>,
  maxUploadBytes: number,
  extensions: readonly string[] = ACCEPTED_EXTENSIONS,
): { accepted: File[]; rejected: RejectedFile[] } {
  const accepted: File[] = [];
  const rejected: RejectedFile[] = [];
  for (const file of files) {
    const extension = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
    if (!file.name.includes('.') || !extensions.includes(extension)) {
      rejected.push({
        filename: file.name,
        message: `${file.name} isn't a supported type (${extensions.join(' ')}).`,
      });
    } else if (file.size > maxUploadBytes) {
      rejected.push({
        filename: file.name,
        message: `${file.name} is too large (${formatBytes(file.size)}). Maximum is ${formatBytes(maxUploadBytes)}.`,
      });
    } else {
      accepted.push(file);
    }
  }
  return { accepted, rejected };
}
