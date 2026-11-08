import { useMemo, useState } from 'react';
import { endOfDayFromDateInput, fromDateInput, toDateInput } from '../utils/dates';
import { MAX_SCOPE_FILENAMES } from '../utils/limits';
import { SCOPE_PRESET_NAME_MAX_CHARS, type ScopePreset } from '../utils/scopePresets';

// The most tags and files one question may be scoped to (api/schemas.py's
// MAX_TAGS_PER_DOCUMENT and REQUEST_FILENAMES_MAX): one more was a 422 on every
// question and every passage search, with nothing saying why.
const MAX_SCOPE_TAGS = 20;

function pageInput(value: string): number | undefined {
  const parsed = Number.parseInt(value, 10);
  return Number.isInteger(parsed) && parsed >= 1 ? parsed : undefined;
}

interface ScopePopoverProps {
  indexedFilenames: string[];
  selectedFilenames: string[];
  onSelectedFilenamesChange: (filenames: string[]) => void;
  availableTags: string[];
  selectedTags: string[];
  onSelectedTagsChange?: (tags: string[]) => void;
  pageRange?: { min?: number; max?: number };
  onPageRangeChange?: (range: { min?: number; max?: number }) => void;
  uploadedAfter?: number;
  onUploadedAfterChange?: (value: number | undefined) => void;
  uploadedBefore?: number;
  onUploadedBeforeChange?: (value: number | undefined) => void;
  scopePresets: ScopePreset[];
  onApplyScopePreset?: (preset: ScopePreset) => void;
  onSaveScopePreset?: (name: string) => void;
  onDeleteScopePreset?: (name: string) => void;
}

/** The composer's scope popover: which documents, tags, pages and upload dates the
 *  conversation's questions search, and the saved scopes. Mounted while open, so its
 *  filter and the preset name start empty each time. */
export default function ScopePopover({
  indexedFilenames,
  selectedFilenames,
  onSelectedFilenamesChange,
  availableTags,
  selectedTags,
  onSelectedTagsChange,
  pageRange,
  onPageRangeChange,
  uploadedAfter,
  onUploadedAfterChange,
  uploadedBefore,
  onUploadedBeforeChange,
  scopePresets,
  onApplyScopePreset,
  onSaveScopePreset,
  onDeleteScopePreset,
}: ScopePopoverProps) {
  const [scopeQuery, setScopeQuery] = useState('');
  const [presetName, setPresetName] = useState<string | null>(null);
  const scopeNarrows =
    selectedFilenames.length > 0 ||
    selectedTags.length > 0 ||
    pageRange?.min !== undefined ||
    pageRange?.max !== undefined ||
    uploadedAfter !== undefined ||
    uploadedBefore !== undefined;
  const selectedFileSet = useMemo(() => new Set(selectedFilenames), [selectedFilenames]);
  const selectedTagSet = useMemo(() => new Set(selectedTags), [selectedTags]);

  const toggleTag = (tag: string) => {
    onSelectedTagsChange?.(
      selectedTagSet.has(tag)
        ? selectedTags.filter((name) => name !== tag)
        : [...selectedTags, tag],
    );
  };

  const toggleFilename = (filename: string) => {
    onSelectedFilenamesChange(
      selectedFileSet.has(filename)
        ? selectedFilenames.filter((name) => name !== filename)
        : [...selectedFilenames, filename],
    );
  };

  // A search hiding an already-selected filename would make it permanently
  // unselectable (the checkbox list is the only way to remove it from scope), so a
  // selected filename always stays visible regardless of match.
  const trimmedScopeQuery = scopeQuery.trim().toLowerCase();
  const visibleFilenames =
    trimmedScopeQuery === ''
      ? indexedFilenames
      : indexedFilenames.filter(
          (filename) =>
            filename.toLowerCase().includes(trimmedScopeQuery) ||
            selectedFileSet.has(filename),
        );

  const tagsFull = selectedTags.length >= MAX_SCOPE_TAGS;
  const filenamesFull = selectedFilenames.length >= MAX_SCOPE_FILENAMES;

  return (
    <div
      className="composer__popover"
      id="composer-scope-popover"
      role="group"
      aria-label="Search scope"
      data-testid="composer-scope-popover"
    >
      <label className="composer__popover-item">
        <input
          type="checkbox"
          checked={selectedFilenames.length === 0 && selectedTags.length === 0}
          onChange={() => {
            onSelectedFilenamesChange([]);
            onSelectedTagsChange?.([]);
          }}
          data-testid="scope-all"
        />
        <span>All documents</span>
      </label>
      {onApplyScopePreset && (scopePresets.length > 0 || onSaveScopePreset) && (
        <div className="composer__presets" role="group" aria-label="Saved scopes">
          {scopePresets.map((preset) => (
            <span key={preset.name} className="composer__preset" data-testid="scope-preset">
              <button
                type="button"
                className="composer__preset-apply"
                onClick={() => onApplyScopePreset(preset)}
                title="Search these documents"
              >
                {preset.name}
              </button>
              {onDeleteScopePreset && (
                <button
                  type="button"
                  className="composer__preset-delete"
                  onClick={() => onDeleteScopePreset(preset.name)}
                  aria-label={`Delete the saved scope ${preset.name}`}
                  data-testid="scope-preset-delete"
                >
                  ✕
                </button>
              )}
            </span>
          ))}
          {onSaveScopePreset &&
            (presetName === null ? (
              <button
                type="button"
                className="composer__preset-save"
                onClick={() => setPresetName('')}
                disabled={!scopeNarrows}
                title={scopeNarrows ? 'Save this scope to pick it again' : 'Choose documents, tags, pages or dates first'}
                data-testid="scope-preset-save"
              >
                Save scope as…
              </button>
            ) : (
              <form
                className="composer__preset-form"
                onSubmit={(event) => {
                  event.preventDefault();
                  if (!presetName.trim()) return;
                  onSaveScopePreset(presetName);
                  setPresetName(null);
                }}
              >
                <input
                  type="text"
                  value={presetName}
                  maxLength={SCOPE_PRESET_NAME_MAX_CHARS}
                  onChange={(event) => setPresetName(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Escape') {
                      event.stopPropagation();
                      setPresetName(null);
                    }
                  }}
                  placeholder="Name"
                  aria-label="Name for this scope"
                  autoFocus
                  data-testid="scope-preset-name"
                />
                <button type="submit" disabled={!presetName.trim()} data-testid="scope-preset-confirm">
                  Save
                </button>
              </form>
            ))}
        </div>
      )}
      {indexedFilenames.length > 5 && (
        <input
          type="search"
          className="composer__popover-search"
          placeholder="Filter documents…"
          aria-label="Filter documents"
          value={scopeQuery}
          onChange={(event) => setScopeQuery(event.target.value)}
          data-testid="composer-scope-search"
        />
      )}
      {tagsFull && (
        <p className="composer__popover-note" data-testid="scope-tags-full">
          Up to {MAX_SCOPE_TAGS} tags per question.
        </p>
      )}
      {filenamesFull && (
        <p className="composer__popover-note" data-testid="scope-files-full">
          Up to {MAX_SCOPE_FILENAMES} documents per question.
        </p>
      )}
      {availableTags.length > 0 && onSelectedTagsChange && (
        <div className="composer__popover-tags" role="group" aria-label="Tags">
          {availableTags.map((tag) => (
            <label key={tag} className="composer__popover-item" data-testid="scope-tag">
              <input
                type="checkbox"
                checked={selectedTagSet.has(tag)}
                disabled={tagsFull && !selectedTagSet.has(tag)}
                onChange={() => toggleTag(tag)}
              />
              <span>#{tag}</span>
            </label>
          ))}
        </div>
      )}
      {onPageRangeChange && selectedFilenames.length === 1 && (
        <div className="composer__popover-pages" data-testid="scope-pages">
          <span>Pages</span>
          <input
            type="number"
            min={1}
            inputMode="numeric"
            value={pageRange?.min ?? ''}
            onChange={(event) =>
              onPageRangeChange({ ...pageRange, min: pageInput(event.target.value) })
            }
            aria-label={`First page of ${selectedFilenames[0]}`}
            placeholder="from"
            data-testid="scope-page-min"
          />
          <span>to</span>
          <input
            type="number"
            min={pageRange?.min ?? 1}
            inputMode="numeric"
            value={pageRange?.max ?? ''}
            onChange={(event) =>
              onPageRangeChange({ ...pageRange, max: pageInput(event.target.value) })
            }
            aria-label={`Last page of ${selectedFilenames[0]}`}
            placeholder="to"
            data-testid="scope-page-max"
          />
          {pageRange?.min !== undefined &&
            pageRange.max !== undefined &&
            pageRange.min > pageRange.max && (
              <span className="composer__popover-note" role="status" data-testid="scope-pages-order">
                Read as pages {pageRange.max}–{pageRange.min}.
              </span>
            )}
        </div>
      )}
      {onUploadedAfterChange && (
        <label className="composer__popover-pages" data-testid="scope-uploaded">
          <span>Uploaded since</span>
          <input
            type="date"
            value={toDateInput(uploadedAfter)}
            onChange={(event) => onUploadedAfterChange(fromDateInput(event.target.value))}
            data-testid="scope-uploaded-after"
          />
        </label>
      )}
      {onUploadedBeforeChange && (
        <label className="composer__popover-pages" data-testid="scope-uploaded-up-to">
          <span>Uploaded up to</span>
          <input
            type="date"
            value={toDateInput(uploadedBefore)}
            onChange={(event) =>
              onUploadedBeforeChange(endOfDayFromDateInput(event.target.value))
            }
            data-testid="scope-uploaded-before"
          />
        </label>
      )}
      {visibleFilenames.map((filename) => (
        <label key={filename} className="composer__popover-item" data-testid="scope-item">
          <input
            type="checkbox"
            checked={selectedFileSet.has(filename)}
            disabled={filenamesFull && !selectedFileSet.has(filename)}
            onChange={() => toggleFilename(filename)}
          />
          <span className="composer__popover-filename mono">{filename}</span>
        </label>
      ))}
    </div>
  );
}
