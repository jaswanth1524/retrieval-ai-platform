/** The full chat history in IndexedDB.
 *
 * localStorage holds ~5 MB per origin, and passage text filled it after a few hundred
 * answers — from then on only the active conversation survived a reload. IndexedDB has
 * a quota in the hundreds of megabytes, so the complete history (source text included)
 * lives here. localStorage keeps a compact copy for an instant first paint and for
 * other tabs' `storage` events, which IndexedDB doesn't fire.
 *
 * Every call degrades to a no-op (load resolves null) where IndexedDB is missing or
 * blocked — private windows in some browsers, test environments.
 */

const DB_NAME = 'docrag';
const STORE = 'chat';
const KEY = 'history';

let connection: Promise<IDBDatabase> | null = null;

export function chatStoreAvailable(): boolean {
  try {
    return typeof indexedDB !== 'undefined' && indexedDB !== null;
  } catch {
    return false;
  }
}

function open(): Promise<IDBDatabase> {
  if (!connection) {
    connection = new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, 1);
      request.onupgradeneeded = () => request.result.createObjectStore(STORE);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
      request.onblocked = () => reject(new Error('IndexedDB open blocked'));
    });
    // A failed open is retried next time rather than cached forever.
    connection.catch(() => {
      connection = null;
    });
  }
  return connection;
}

export async function loadChatStore(): Promise<unknown> {
  if (!chatStoreAvailable()) return null;
  try {
    const db = await open();
    return await new Promise<unknown>((resolve, reject) => {
      const request = db.transaction(STORE, 'readonly').objectStore(STORE).get(KEY);
      request.onsuccess = () => resolve(request.result ?? null);
      request.onerror = () => reject(request.error);
    });
  } catch {
    return null;
  }
}

/** Resolves true once the value is durably written, false when it couldn't be. */
export async function saveChatStore(value: unknown): Promise<boolean> {
  if (!chatStoreAvailable()) return false;
  try {
    const db = await open();
    return await new Promise<boolean>((resolve) => {
      const transaction = db.transaction(STORE, 'readwrite');
      transaction.objectStore(STORE).put(value, KEY);
      transaction.oncomplete = () => resolve(true);
      transaction.onerror = () => resolve(false);
      transaction.onabort = () => resolve(false);
    });
  } catch {
    return false;
  }
}

/** Test seam: drop the cached connection (e.g. after swapping in a fresh fake). */
export function resetChatStoreConnection(): void {
  connection = null;
}
