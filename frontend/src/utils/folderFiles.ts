/** Folders handed to the app — dropped, or picked with the folder input — as the
 *  documents in them. The server indexes files by name only, so a folder upload keeps
 *  one file per name and leaves out what it can't index, instead of failing on it. */

/** The most files one folder upload sends; the rest are reported, not sent. */
export const MAX_FOLDER_FILES = 500;

// The parts of the File and Directory Entries API used here (webkitGetAsEntry), so
// tests can hand in plain objects.
export interface EntryLike {
  isFile: boolean;
  isDirectory: boolean;
  name: string;
  file?: (success: (file: File) => void, failure?: (error: unknown) => void) => void;
  createReader?: () => {
    readEntries: (success: (entries: EntryLike[]) => void, failure?: (error: unknown) => void) => void;
  };
}

/** The dropped items as entries, read now: the browser empties the drop's item list
 *  once the drop handler returns, so this must run synchronously inside it. Null when
 *  the browser has no entries API (or nothing was dropped). */
export function droppedEntries(dataTransfer: DataTransfer | null): EntryLike[] | null {
  const items = Array.from(dataTransfer?.items ?? []);
  const entries: EntryLike[] = [];
  for (const item of items) {
    const entry = typeof item.webkitGetAsEntry === 'function' ? item.webkitGetAsEntry() : null;
    if (entry) entries.push(entry as unknown as EntryLike);
  }
  return entries.length > 0 ? entries : null;
}

export function hasFolder(entries: EntryLike[] | null): boolean {
  return entries?.some((entry) => entry.isDirectory) ?? false;
}

function readAll(directory: EntryLike): Promise<EntryLike[]> {
  const reader = directory.createReader?.();
  if (!reader) return Promise.resolve([]);
  const all: EntryLike[] = [];
  // readEntries answers in batches (100 at a time in Chrome) until it returns none.
  return new Promise((resolve) => {
    const next = () =>
      reader.readEntries(
        (batch) => {
          if (batch.length === 0) {
            resolve(all);
            return;
          }
          all.push(...batch);
          next();
        },
        () => resolve(all),
      );
    next();
  });
}

function fileOf(entry: EntryLike): Promise<File | null> {
  return new Promise((resolve) => {
    if (!entry.file) {
      resolve(null);
      return;
    }
    entry.file(resolve, () => resolve(null));
  });
}

/** Every file under ``entries`` (folders walked, hidden files and folders left out),
 *  stopping once ``limit`` + 1 are found — enough to know the folder was cut. */
export async function filesUnder(entries: EntryLike[], limit = MAX_FOLDER_FILES): Promise<File[]> {
  const files: File[] = [];
  const queue = [...entries];
  while (queue.length > 0 && files.length <= limit) {
    const entry = queue.shift()!;
    if (entry.name.startsWith('.')) continue;
    if (entry.isDirectory) {
      queue.push(...(await readAll(entry)));
    } else if (entry.isFile) {
      const file = await fileOf(entry);
      if (file) files.push(file);
    }
  }
  return files;
}

export interface FolderPick {
  /** One per name, of a type the server indexes, at most ``MAX_FOLDER_FILES``. */
  files: File[];
  /** Files of other types (and hidden ones) left out. */
  otherTypes: number;
  /** Names already taken by another file of the folder (only the first is sent). */
  duplicates: string[];
  /** More files than ``MAX_FOLDER_FILES``: the rest weren't sent. */
  truncated: boolean;
}

/** Which of a folder's files to upload. */
export function pickFolderFiles(
  files: Iterable<File>,
  extensions: readonly string[],
  limit = MAX_FOLDER_FILES,
): FolderPick {
  const picked: File[] = [];
  const seen = new Set<string>();
  const duplicates: string[] = [];
  let otherTypes = 0;
  let truncated = false;
  for (const file of files) {
    const relative = (file as File & { webkitRelativePath?: string }).webkitRelativePath ?? '';
    const hidden = file.name.startsWith('.') || relative.split('/').some((part) => part.startsWith('.'));
    const dot = file.name.lastIndexOf('.');
    const extension = dot > 0 ? file.name.slice(dot).toLowerCase() : '';
    if (hidden || !extensions.includes(extension)) {
      otherTypes += 1;
      continue;
    }
    const key = file.name.toLowerCase();
    if (seen.has(key)) {
      duplicates.push(file.name);
      continue;
    }
    if (picked.length >= limit) {
      truncated = true;
      break;
    }
    seen.add(key);
    picked.push(file);
  }
  return { files: picked, otherTypes, duplicates, truncated };
}

/** What a folder upload left out, in one line; null when it sent everything. */
export function folderPickNote(pick: FolderPick): string | null {
  const parts: string[] = [];
  if (pick.otherTypes > 0) {
    parts.push(`${pick.otherTypes} file${pick.otherTypes === 1 ? '' : 's'} of other types`);
  }
  if (pick.duplicates.length > 0) {
    parts.push(
      `${pick.duplicates.length} with a name already used in the folder (${pick.duplicates.slice(0, 3).join(', ')}${pick.duplicates.length > 3 ? '…' : ''})`,
    );
  }
  if (pick.truncated) parts.push(`everything past the first ${MAX_FOLDER_FILES}`);
  return parts.length > 0 ? `Left out ${parts.join(', ')}.` : null;
}
