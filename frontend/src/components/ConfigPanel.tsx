import { useState } from 'react';
import { getApiKey, setApiKey } from '../api/client';
import type { PublicConfigResponse, QuestionOverrides } from '../api/types';
import './ConfigPanel.css';

interface ConfigPanelProps {
  config: PublicConfigResponse;
  overrides: QuestionOverrides;
  onOverridesChange: (value: QuestionOverrides) => void;
  disabled?: boolean;
}

const READONLY_METRICS: Array<[label: string, key: keyof PublicConfigResponse]> = [
  ['RRF k', 'rrf_k'],
  ['Fused top N', 'fused_top_n'],
  ['Min rerank score', 'rerank_min_score'],
];

const IDENTIFIER_FIELDS: Array<[label: string, key: keyof PublicConfigResponse]> = [
  ['Qdrant collection', 'qdrant_collection'],
  ['Dense embedding model', 'dense_embedding_model'],
  ['Sparse embedding model', 'sparse_embedding_model'],
  ['Reranker model', 'reranker_model'],
  ['Embedding model tag', 'embedding_model_tag'],
  ['LLM provider', 'llm_provider'],
  ['LLM model', 'llm_model'],
  ['OpenAI model', 'openai_model'],
  ['Dense retrieval limit', 'dense_retrieval_limit'],
  ['Sparse retrieval limit', 'sparse_retrieval_limit'],
];

const EMPTY_OVERRIDES: QuestionOverrides = {
  rerankTopK: null,
  maxContextChunks: null,
  llmTemperature: null,
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

interface StepperCardProps {
  label: string;
  displayValue: string;
  onStep: (delta: number) => void;
  disabled?: boolean;
}

function StepperCard({ label, displayValue, onStep, disabled }: StepperCardProps) {
  return (
    <div className="config-panel__metric config-panel__metric--stepper">
      <div>
        <span className="config-panel__metric-value mono" aria-live="polite">
          {displayValue}
        </span>
        <span className="config-panel__metric-label">{label}</span>
      </div>
      <div className="config-panel__stepper-controls">
        <button
          type="button"
          className="config-panel__stepper-btn"
          onClick={() => onStep(1)}
          disabled={disabled}
          aria-label={`Increase ${label}`}
        >
          &#9650;
        </button>
        <button
          type="button"
          className="config-panel__stepper-btn"
          onClick={() => onStep(-1)}
          disabled={disabled}
          aria-label={`Decrease ${label}`}
        >
          &#9660;
        </button>
      </div>
    </div>
  );
}

function ConfigPanel({ config, overrides, onOverridesChange, disabled }: ConfigPanelProps) {
  const [expanded, setExpanded] = useState(false);
  // Only meaningful when the server has API_KEY set (see api/settings.py) — with no
  // key configured server-side, sending this header is harmless and ignored.
  const [apiKeyDraft, setApiKeyDraft] = useState(getApiKey);

  const rerankTopK = overrides.rerankTopK ?? config.rerank_top_k;
  const maxContextChunks = overrides.maxContextChunks ?? config.max_context_chunks;
  const llmTemperature = overrides.llmTemperature ?? config.llm_temperature;

  const stepRerankTopK = (delta: number) => {
    const next = clamp(rerankTopK + delta, 1, config.rerank_top_k_limit);
    onOverridesChange({ ...overrides, rerankTopK: next });
  };

  const stepMaxContextChunks = (delta: number) => {
    const next = clamp(maxContextChunks + delta, 1, config.max_context_chunks_limit);
    onOverridesChange({ ...overrides, maxContextChunks: next });
  };

  const stepTemperature = (direction: 1 | -1) => {
    // Guard against binary float drift (0.1 + 0.2 !== 0.3) across repeated clicks.
    const next = clamp(
      Math.round((llmTemperature + direction * 0.1) * 10) / 10,
      0,
      config.llm_temperature_max,
    );
    onOverridesChange({ ...overrides, llmTemperature: next });
  };

  return (
    <div className="config-panel">
      <div className="config-panel__section-label">Retrieval settings</div>
      <div className="config-panel__metrics">
        {READONLY_METRICS.map(([label, key]) => (
          <div key={key} className="config-panel__metric">
            <span className="config-panel__metric-value mono">{config[key]}</span>
            <span className="config-panel__metric-label">{label}</span>
          </div>
        ))}
        <StepperCard
          label="Rerank top K"
          displayValue={String(rerankTopK)}
          onStep={stepRerankTopK}
          disabled={disabled}
        />
        <StepperCard
          label="Context chunks"
          displayValue={String(maxContextChunks)}
          onStep={stepMaxContextChunks}
          disabled={disabled}
        />
      </div>

      <div className="config-panel__advanced-label">Advanced</div>
      <StepperCard
        label="Temperature"
        displayValue={llmTemperature.toFixed(1)}
        onStep={(delta) => stepTemperature(delta > 0 ? 1 : -1)}
        disabled={disabled}
      />
      <button
        type="button"
        className="config-panel__reset"
        onClick={() => onOverridesChange(EMPTY_OVERRIDES)}
        disabled={disabled}
      >
        Reset to defaults
      </button>

      <label className="config-panel__api-key-label" htmlFor="config-panel-api-key">
        API key (only needed if the server has one configured)
      </label>
      <input
        id="config-panel-api-key"
        type="password"
        className="config-panel__api-key-input"
        value={apiKeyDraft}
        onChange={(event) => {
          setApiKeyDraft(event.target.value);
          setApiKey(event.target.value);
        }}
        placeholder="X-API-Key"
        data-testid="config-panel-api-key"
      />

      <button
        type="button"
        className="config-panel__toggle"
        onClick={() => setExpanded((prev) => !prev)}
      >
        {expanded ? 'Hide system info' : 'System info'}
      </button>
      {expanded && (
        <table className="config-panel__table">
          <tbody>
            {IDENTIFIER_FIELDS.map(([label, key]) => (
              <tr key={key}>
                <td>{label}</td>
                <td className="mono">{config[key]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default ConfigPanel;
