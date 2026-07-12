import './DocumentFilter.css';

interface DocumentFilterProps {
  filenames: string[];
  selected: string[];
  onChange: (selected: string[]) => void;
  disabled?: boolean;
}

function DocumentFilter({ filenames, selected, onChange, disabled }: DocumentFilterProps) {
  if (filenames.length === 0) return null;

  const toggle = (filename: string) => {
    onChange(
      selected.includes(filename)
        ? selected.filter((name) => name !== filename)
        : [...selected, filename],
    );
  };

  return (
    <div className="document-filter">
      <div className="document-filter__label">Search scope</div>
      <label className="document-filter__item">
        <input
          type="checkbox"
          checked={selected.length === 0}
          onChange={() => onChange([])}
          disabled={disabled}
          data-testid="document-filter-all"
        />
        <span>All documents</span>
      </label>
      {filenames.map((filename) => (
        <label key={filename} className="document-filter__item" data-testid="document-filter-item">
          <input
            type="checkbox"
            checked={selected.includes(filename)}
            onChange={() => toggle(filename)}
            disabled={disabled}
          />
          <span className="document-filter__filename mono">{filename}</span>
        </label>
      ))}
    </div>
  );
}

export default DocumentFilter;
