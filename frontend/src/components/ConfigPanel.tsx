import { useState } from 'react';
import type { PublicConfigResponse } from '../api/types';
import './ConfigPanel.css';

interface ConfigPanelProps {
  config: PublicConfigResponse;
}

const METRICS: Array<[label: string, key: keyof PublicConfigResponse]> = [
  ['RRF k', 'rrf_k'],
  ['Fused top N', 'fused_top_n'],
  ['Rerank top K', 'rerank_top_k'],
  ['Context chunks', 'max_context_chunks'],
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

function ConfigPanel({ config }: ConfigPanelProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="config-panel">
      <div className="config-panel__section-label">Retrieval settings</div>
      <div className="config-panel__metrics">
        {METRICS.map(([label, key]) => (
          <div key={key} className="config-panel__metric">
            <span className="config-panel__metric-value mono">{config[key]}</span>
            <span className="config-panel__metric-label">{label}</span>
          </div>
        ))}
      </div>
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
