import { useState } from 'react';
import { getApiKey, setApiKey } from '../api/client';
import type { PublicConfigResponse, QuestionOverrides } from '../api/types';
import { useDialog } from '../hooks/useDialog';
import './SettingsModal.css';

interface SettingsModalProps {
  config: PublicConfigResponse;
  overrides: QuestionOverrides;
  onOverridesChange: (value: QuestionOverrides) => void;
  onClose: () => void;
  onSaved: () => void;
  disabled?: boolean;
}

const EMPTY_OVERRIDES: QuestionOverrides = {
  rerankTopK: null,
  maxContextChunks: null,
  llmTemperature: null,
};

const SYSTEM_FIELDS: Array<[label: string, key: keyof PublicConfigResponse]> = [
  ['Qdrant collection', 'qdrant_collection'],
  ['Dense embedding model', 'dense_embedding_model'],
  ['Sparse embedding model', 'sparse_embedding_model'],
  ['Reranker model', 'reranker_model'],
  ['Embedding model tag', 'embedding_model_tag'],
  ['RRF k', 'rrf_k'],
  ['Fused top N', 'fused_top_n'],
  ['Min rerank score', 'rerank_min_score'],
  ['Dense retrieval limit', 'dense_retrieval_limit'],
  ['Sparse retrieval limit', 'sparse_retrieval_limit'],
];

interface SliderRowProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format: (value: number) => string;
  onChange: (value: number) => void;
  disabled?: boolean;
  testId: string;
}

function SliderRow({
  label,
  value,
  min,
  max,
  step,
  format,
  onChange,
  disabled,
  testId,
}: SliderRowProps) {
  return (
    <label className="settings__slider-row">
      <span className="settings__slider-label">{label}</span>
      <input
        type="range"
        className="settings__slider"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
        data-testid={testId}
      />
      <span className="settings__slider-value mono">{format(value)}</span>
    </label>
  );
}

function SettingsModal({
  config,
  overrides,
  onOverridesChange,
  onClose,
  onSaved,
  disabled,
}: SettingsModalProps) {
  const dialogRef = useDialog(true, onClose);
  const [apiKeyDraft, setApiKeyDraft] = useState(getApiKey);
  // Staged like the API key: sliders only touch this draft, never onOverridesChange
  // directly, so a drag doesn't localStorage-write on every tick and Cancel (unmount,
  // since the modal is only ever conditionally rendered) discards it for free.
  const [draftOverrides, setDraftOverrides] = useState(overrides);

  const rerankTopK = draftOverrides.rerankTopK ?? config.rerank_top_k;
  const maxContextChunks = draftOverrides.maxContextChunks ?? config.max_context_chunks;
  const llmTemperature = draftOverrides.llmTemperature ?? config.llm_temperature;

  const save = () => {
    onOverridesChange(draftOverrides);
    setApiKey(apiKeyDraft);
    onSaved();
    onClose();
  };

  return (
    <div
      className="settings__backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
      data-testid="settings-backdrop"
    >
      <div
        className="settings"
        role="dialog"
        aria-modal="true"
        aria-label="Settings"
        ref={dialogRef}
        data-testid="settings-modal"
      >
        <header className="settings__header">
          <h2 className="settings__title">Settings</h2>
          <button
            type="button"
            className="settings__close"
            onClick={onClose}
            aria-label="Close settings"
            data-testid="settings-close"
          >
            <span aria-hidden="true">✕</span>
          </button>
        </header>

        <div className="settings__body">
          <section className="settings__section">
            <h3 className="settings__section-title mono">Model</h3>
            <div className="settings__provider-cards">
              <div
                className={`settings__provider${config.ollama_available ? '' : ' settings__provider--off'}`}
              >
                <span className="settings__provider-name">Ollama</span>
                <span className="settings__provider-model mono">{config.llm_model}</span>
                <span className="settings__provider-status mono">
                  {config.ollama_available ? 'reachable' : 'unreachable'}
                </span>
              </div>
              <div
                className={`settings__provider${config.openai_available ? '' : ' settings__provider--off'}`}
              >
                <span className="settings__provider-name">OpenAI</span>
                <span className="settings__provider-model mono">{config.openai_model}</span>
                <span className="settings__provider-status mono">
                  {config.openai_available ? 'key configured' : 'no key'}
                </span>
              </div>
            </div>
          </section>

          <section className="settings__section">
            <h3 className="settings__section-title mono">Retrieval</h3>
            {/* Bounds come from the server's own limits, never hardcoded — a slider
                that can reach a value the backend rejects is a 400 waiting to happen,
                and rerank_top_k_limit in particular is capped by the live fused_top_n. */}
            <SliderRow
              label="Rerank top K"
              value={rerankTopK}
              min={1}
              max={config.rerank_top_k_limit}
              step={1}
              format={String}
              disabled={disabled}
              testId="settings-rerank-top-k"
              onChange={(value) => setDraftOverrides({ ...draftOverrides, rerankTopK: value })}
            />
            <SliderRow
              label="Context chunks"
              value={maxContextChunks}
              min={1}
              max={config.max_context_chunks_limit}
              step={1}
              format={String}
              disabled={disabled}
              testId="settings-max-context-chunks"
              onChange={(value) =>
                setDraftOverrides({ ...draftOverrides, maxContextChunks: value })
              }
            />
            <SliderRow
              label="Temperature"
              value={llmTemperature}
              min={0}
              max={config.llm_temperature_max}
              step={0.1}
              format={(value) => value.toFixed(1)}
              disabled={disabled}
              testId="settings-temperature"
              onChange={(value) =>
                // Guard against binary float drift across repeated drags.
                setDraftOverrides({
                  ...draftOverrides,
                  llmTemperature: Math.round(value * 10) / 10,
                })
              }
            />
            <button
              type="button"
              className="settings__reset"
              onClick={() => setDraftOverrides(EMPTY_OVERRIDES)}
              disabled={disabled}
              data-testid="settings-reset"
            >
              Reset to server defaults
            </button>
          </section>

          <section className="settings__section">
            <h3 className="settings__section-title mono">Access</h3>
            {/* Not in the design, but not prototype UI either: a deployment with
                API_KEY set is unusable from the browser without this field, and every
                /documents call 401s until it is filled in. */}
            <label className="settings__field-label" htmlFor="settings-api-key">
              API key
            </label>
            <input
              id="settings-api-key"
              type="password"
              className="settings__input"
              value={apiKeyDraft}
              onChange={(event) => setApiKeyDraft(event.target.value)}
              placeholder="X-API-Key"
              data-testid="settings-api-key"
            />
            <p className="settings__hint">
              Only needed when the server has <code>API_KEY</code> set. Stored in this browser.
            </p>
          </section>

          <details className="settings__system">
            <summary className="mono">System info</summary>
            <table className="settings__table">
              <tbody>
                {SYSTEM_FIELDS.map(([label, key]) => (
                  <tr key={key}>
                    <td>{label}</td>
                    <td className="mono">{String(config[key])}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        </div>

        <footer className="settings__footer">
          <button type="button" className="settings__cancel" onClick={onClose}>
            Cancel
          </button>
          <button type="button" className="settings__save" onClick={save} data-testid="settings-save">
            Save
          </button>
        </footer>
      </div>
    </div>
  );
}

export default SettingsModal;
