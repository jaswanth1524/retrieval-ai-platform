"""The settings reference (``docs/settings.md``), generated from ``AppSettings``.

Each setting's description is the comment block directly above its field in
``api/settings.py``; ``SECTIONS`` groups them. ``tests/test_settings_docs.py`` fails when
a setting has no comment or no section, or when the file is out of date — regenerate
it with ``make settings-docs`` (``python -m api.settings_docs``).
"""

from __future__ import annotations

import ast
import inspect
import sys
import types
import typing
from pathlib import Path

from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from api.settings import AppSettings

DOCS_PATH = Path(__file__).resolve().parent.parent / "docs" / "settings.md"

# (title, one-line summary, field names) in reading order. Every field is in exactly one.
SECTIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "Vector store",
        "Where Qdrant is and how the collection is laid out.",
        (
            "qdrant_url",
            "qdrant_collection",
            "qdrant_dense_vector_name",
            "qdrant_sparse_vector_name",
            "qdrant_dense_vector_size",
            "qdrant_prefer_grpc",
            "qdrant_grpc_port",
            "qdrant_api_key",
            "qdrant_quantization",
            "qdrant_quantization_oversampling",
        ),
    ),
    (
        "Embedding and reranking models",
        "The local models. Changing an embedding setting needs a re-index.",
        (
            "dense_embedding_model",
            "sparse_embedding_model",
            "dense_query_instruction",
            "dense_passage_prefix",
            "sparse_language",
            "embedding_model_tag",
            "reranker_model",
            "reranker_onnx_file",
            "reranker_batch_size",
            "embedding_threads",
            "reranker_threads",
            "rerank_cache_size",
            "warmup_models",
        ),
    ),
    (
        "Retrieval and ranking",
        "Hybrid search, fusion, reranking and which passages reach the prompt.",
        (
            "rrf_k",
            "dense_retrieval_limit",
            "sparse_retrieval_limit",
            "fused_top_n",
            "rerank_candidates",
            "rerank_top_k",
            "max_context_chunks",
            "rerank_min_score",
            "rerank_fallback_count",
            "rerank_relative_floor",
            "recency_boost_weight",
            "context_neighbor_radius",
            "context_neighbor_max_chars",
            "context_diversity_enabled",
            "context_diversity_max_similarity",
            "query_expansion_enabled",
            "query_expansion_count",
            "query_expansion_max_tokens",
        ),
    ),
    (
        "Generation providers",
        "The LLM that writes answers. Hosted providers are optional add-ons.",
        (
            "llm_provider",
            "allowed_request_providers",
            "llm_model",
            "llm_temperature",
            "llm_max_tokens",
            "llm_request_timeout_seconds",
            "llm_prefill_tokens_per_second",
            "llm_num_retries",
            "llm_context_window",
            "ollama_base_url",
            "ollama_keep_alive",
            "ollama_num_ctx",
            "ollama_max_parallel",
            "openai_api_key",
            "openai_model",
            "openai_reasoning_effort",
            "reasoning_token_headroom",
            "openai_compatible_base_url",
            "openai_compatible_model",
            "openai_compatible_api_key",
            "anthropic_api_key",
            "anthropic_model",
        ),
    ),
    (
        "Answers and conversations",
        "Prompt size, conversation memory, citations, follow-ups and caching.",
        (
            "prompt_token_budget",
            "answer_language",
            "answer_format",
            "conversation_condense_enabled",
            "conversation_max_history_messages",
            "generation_history_assistant_chars",
            "condense_max_tokens",
            "condense_cache_size",
            "citation_retry_enabled",
            "citation_retry_mode",
            "auto_continue_max",
            "followups_enabled",
            "followup_count",
            "followup_max_tokens",
            "answer_cache_size",
            "answer_cache_ttl_seconds",
            "prompt_cache_size",
        ),
    ),
    (
        "Summaries",
        "Whole-document summaries, comparisons and briefs.",
        (
            "summary_batch_chars",
            "summary_max_batches",
            "summary_max_tokens",
            "summary_parallel_parts",
            "summary_max_documents",
        ),
    ),
    (
        "Documents and chunking",
        "Upload limits, parsing limits and how text is cut into passages.",
        (
            "max_upload_bytes",
            "max_import_bytes",
            "max_migrate_bytes",
            "max_archive_members",
            "min_free_bytes",
            "max_chunks_per_document",
            "front_matter_tags",
            "chunk_size_tokens",
            "chunk_overlap_tokens",
            "min_section_words",
            "csv_rows_per_section",
            "max_pdf_pages",
            "max_ocr_pages",
            "max_spreadsheet_cells",
            "parse_timeout_seconds",
            "parse_memory_limit_mb",
        ),
    ),
    (
        "Indexing jobs",
        "The ingest queue, job records and background re-indexing.",
        (
            "ingest_batch_size",
            "ingest_workers",
            "ingest_upsert_batch_size",
            "ingest_reuse_vectors",
            "ingest_max_pending_bytes",
            "ingest_jobs_max_retained",
            "job_store_backend",
            "job_store_path",
            "requeue_interrupted_jobs",
            "auto_reindex_stale",
            "auto_reindex_batch",
            "import_spool_dir",
            "document_listing_ttl_seconds",
        ),
    ),
    (
        "Originals, backups and workspaces",
        "Keeping uploaded files, scheduled backups and separate corpora.",
        (
            "raw_document_dir",
            "raw_document_versions",
            "backup_dir",
            "backup_interval_hours",
            "backup_keep",
            "workspaces_enabled",
            "workspace_settings_path",
        ),
    ),
    (
        "Watched folders",
        "Keeping the index in step with folders on the server.",
        (
            "watch_dir",
            "watch_dirs",
            "watch_interval_seconds",
            "watch_delete_missing",
            "watch_keep_paths",
            "watch_workspace",
            "watch_max_retries",
            "watch_retry_max_seconds",
            "watch_include",
            "watch_exclude",
        ),
    ),
    (
        "Web sources",
        "Documents fetched from URLs, sitemaps and feeds.",
        (
            "url_ingest_enabled",
            "url_ingest_allow_private",
            "url_ingest_timeout_seconds",
            "url_refresh_interval_minutes",
            "url_refresh_batch",
            "url_refresh_state_path",
            "sitemap_max_pages",
        ),
    ),
    (
        "Access and limits",
        "Keys, CORS and how much work callers may start.",
        (
            "api_key",
            "api_read_key",
            "cors_allow_origins",
            "rate_limit_per_minute",
            "max_concurrent_questions",
            "max_concurrent_summaries",
        ),
    ),
    (
        "Logs, traces and feedback",
        "What the server records about its own work and its users' questions.",
        (
            "log_level",
            "log_format",
            "trace_enabled",
            "trace_max_retained",
            "trace_store_path",
            "trace_store_max_rows",
            "trace_leg_ranks",
            "recheck_time_budget_seconds",
            "feedback_enabled",
            "feedback_store_path",
            "feedback_max_rows",
            "query_log_enabled",
            "query_log_path",
            "query_log_max_rows",
            "audit_log_path",
            "audit_log_max_rows",
        ),
    ),
    (
        "Notifications",
        "A webhook told when unattended work fails.",
        (
            "notify_url",
            "notify_events",
            "notify_format",
            "notify_allow_private",
            "notify_min_interval_seconds",
        ),
    ),
)


def field_comments() -> dict[str, str]:
    """Each field's comment block (the ``#`` lines directly above it), as one paragraph;
    "" for a field with none."""

    source = inspect.getsource(AppSettings)
    lines = source.splitlines()
    class_node = ast.parse(source).body[0]
    assert isinstance(class_node, ast.ClassDef)
    comments: dict[str, str] = {}
    for node in class_node.body:
        if not (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)):
            continue
        block: list[str] = []
        index = node.lineno - 2  # the line above the field (lineno is 1-based)
        while index >= 0 and lines[index].strip().startswith("#"):
            block.append(lines[index].strip().removeprefix("#").strip())
            index -= 1
        comments[node.target.id] = " ".join(line for line in reversed(block) if line)
    comments.pop("model_config", None)
    return comments


def _type_name(annotation: object) -> str:
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Literal:
        return " or ".join(f'"{arg}"' if isinstance(arg, str) else str(arg) for arg in args)
    if origin in (typing.Union, types.UnionType):
        return " or ".join(_type_name(arg) for arg in args)
    if origin is typing.Annotated:
        return _type_name(args[0])
    if origin is list:
        return f"list of {_type_name(args[0])}" if args else "list"
    if annotation is type(None):
        return "unset"
    name = getattr(annotation, "__name__", str(annotation))
    plain = {"str": "text", "int": "integer", "float": "number", "bool": "true/false"}
    return plain.get(name, name)


def _default(field: FieldInfo) -> str:
    if field.default_factory is not None:
        value: object = field.default_factory()  # type: ignore[call-arg]
    elif field.default is PydanticUndefined:
        return "required"
    else:
        value = field.default
    if value is None:
        return "unset"
    if value == "":
        return "empty"
    if isinstance(value, bool):
        return "true" if value else "false"
    return f"`{value}`"


def _limits(field: FieldInfo) -> str:
    bounds: list[str] = []
    for item in field.metadata:
        for attr, word in (("ge", "≥"), ("gt", ">"), ("le", "≤"), ("lt", "<")):
            value = getattr(item, attr, None)
            if value is not None:
                bounds.append(f"{word} {value}")
        max_length = getattr(item, "max_length", None)
        if max_length is not None:
            bounds.append(f"at most {max_length} characters")
    return ", ".join(bounds)


def _markdown(text: str) -> str:
    """A comment as Markdown text: ``<name>`` would vanish as an HTML tag and ``*.py``
    could start emphasis."""

    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("*", "\\*")


def render() -> str:
    fields = AppSettings.model_fields
    comments = field_comments()
    out = [
        "# Settings reference",
        "",
        "<!-- Generated from api/settings.py by `make settings-docs`; edit the comments there. -->",
        "",
        "Every setting is an environment variable (or a line in `.env`): the name below,",
        "upper-cased. `.env.example` lists them all at their defaults. Settings marked",
        "*per workspace* can also be changed for one workspace in Settings → Administration.",
        "",
    ]
    for title, _summary, _names in SECTIONS:
        anchor = title.lower().replace(",", "").replace(" ", "-")
        out.append(f"- [{title}](#{anchor})")
    out.append("")
    workspace_fields = _workspace_fields()
    for title, summary, names in SECTIONS:
        out += [f"## {title}", "", summary, ""]
        for name in names:
            field = fields[name]
            facts = [_type_name(field.annotation), f"default {_default(field)}"]
            if limits := _limits(field):
                facts.append(limits)
            if name in workspace_fields:
                facts.append("*per workspace*")
            out.append(f"- **`{name.upper()}`** ({'; '.join(facts)}) — {_markdown(comments[name])}")
        out.append("")
    return "\n".join(out)


def _workspace_fields() -> frozenset[str]:
    from api.workspace_settings import WORKSPACE_SETTING_FIELDS

    return frozenset(WORKSPACE_SETTING_FIELDS)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0]) if args else DOCS_PATH
    path.write_text(render(), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
