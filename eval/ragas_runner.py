"""Optional RAGAS evaluation runner for DocRAG answers."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_METRICS: tuple[str, ...] = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


class EvaluationDataError(ValueError):
    """Raised when an evaluation dataset is malformed."""


class RagasUnavailableError(RuntimeError):
    """Raised when optional RAGAS runtime dependencies are unavailable."""


@dataclass(frozen=True)
class EvaluationExample:
    """One single-turn RAG evaluation example."""

    question: str
    answer: str
    contexts: list[str]
    reference: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> EvaluationExample:
        """Build an example from JSON-compatible data."""

        question = required_string(data, ("question", "user_input"))
        answer = required_string(data, ("answer", "response"))
        contexts = required_string_list(data, ("contexts", "retrieved_contexts"))
        reference = optional_string(data, ("reference", "ground_truth"))
        return cls(
            question=question,
            answer=answer,
            contexts=contexts,
            reference=reference,
        )

    def to_ragas_row(self) -> dict[str, Any]:
        """Return the single-turn RAGAS row shape."""

        row: dict[str, Any] = {
            "user_input": self.question,
            "response": self.answer,
            "retrieved_contexts": self.contexts,
        }
        if self.reference is not None:
            row["reference"] = self.reference
        return row


def load_examples(path: Path) -> list[EvaluationExample]:
    """Load evaluation examples from JSON or JSONL."""

    if path.suffix.lower() == ".jsonl":
        return load_jsonl_examples(path)
    if path.suffix.lower() == ".json":
        return load_json_examples(path)
    raise EvaluationDataError("Evaluation dataset must be a .json or .jsonl file.")


def load_jsonl_examples(path: Path) -> list[EvaluationExample]:
    """Load examples from newline-delimited JSON."""

    examples: list[EvaluationExample] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationDataError(f"Invalid JSON on line {line_number}.") from exc
        if not isinstance(data, dict):
            raise EvaluationDataError(f"Line {line_number} must be a JSON object.")
        examples.append(EvaluationExample.from_mapping(data))
    return require_examples(examples)


def load_json_examples(path: Path) -> list[EvaluationExample]:
    """Load examples from a JSON list or an object with an examples key."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationDataError("Evaluation dataset is not valid JSON.") from exc

    if isinstance(data, dict):
        data = data.get("examples")
    if not isinstance(data, list):
        raise EvaluationDataError("JSON evaluation dataset must be a list or contain examples.")

    examples: list[EvaluationExample] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise EvaluationDataError(f"Example {index} must be a JSON object.")
        examples.append(EvaluationExample.from_mapping(item))
    return require_examples(examples)


def run_ragas_evaluation(
    examples: Sequence[EvaluationExample],
    *,
    metric_names: Sequence[str] = DEFAULT_METRICS,
    evaluate_func: Callable[..., Any] | None = None,
    dataset_factory: Callable[[list[dict[str, Any]]], Any] | None = None,
    metric_objects: Sequence[Any] | None = None,
) -> Any:
    """Run RAGAS evaluation on prepared examples."""

    rows = [example.to_ragas_row() for example in require_examples(list(examples))]

    if evaluate_func is None or dataset_factory is None or metric_objects is None:
        evaluate_func, dataset_factory, metric_objects = load_ragas_runtime(metric_names)

    dataset = dataset_factory(rows)
    return evaluate_func(
        dataset=dataset,
        metrics=list(metric_objects),
        raise_exceptions=False,
        show_progress=False,
    )


def load_ragas_runtime(
    metric_names: Sequence[str],
) -> tuple[Callable[..., Any], Callable[[list[dict[str, Any]]], Any], list[Any]]:
    """Lazy-load RAGAS, datasets, and requested metric objects."""

    try:
        datasets_module = importlib.import_module("datasets")
        ragas_module = importlib.import_module("ragas")
        metrics_module = importlib.import_module("ragas.metrics")
    except ImportError as exc:
        raise RagasUnavailableError(
            "RAGAS evaluation dependencies are not importable. "
            "Install the eval extra and verify the RAGAS/LangChain versions are compatible."
        ) from exc

    evaluate_func = getattr(ragas_module, "evaluate", None)
    dataset_class = getattr(datasets_module, "Dataset", None)
    if not callable(evaluate_func) or dataset_class is None:
        raise RagasUnavailableError("RAGAS runtime did not expose evaluate or Dataset.")

    metrics = []
    for metric_name in metric_names:
        try:
            metrics.append(getattr(metrics_module, metric_name))
        except AttributeError as exc:
            raise RagasUnavailableError(f"Unknown RAGAS metric '{metric_name}'.") from exc

    return evaluate_func, dataset_class.from_list, metrics


def result_to_jsonable(result: Any) -> dict[str, Any]:
    """Convert common RAGAS result objects into JSON-serializable data."""

    if isinstance(result, dict):
        return result

    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, dict):
            return converted

    to_pandas = getattr(result, "to_pandas", None)
    if callable(to_pandas):
        dataframe = to_pandas()
        to_records = getattr(dataframe, "to_dict", None)
        if callable(to_records):
            return {"rows": to_records(orient="records")}

    return {"result": str(result)}


def parse_metric_names(raw_metrics: str) -> tuple[str, ...]:
    """Parse a comma-separated metric list."""

    metrics = tuple(metric.strip() for metric in raw_metrics.split(",") if metric.strip())
    if not metrics:
        raise EvaluationDataError("At least one metric must be requested.")
    return metrics


def required_string(data: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    """Read a required non-empty string field from one of several aliases."""

    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise EvaluationDataError(f"Missing required string field '{keys[0]}'.")


def optional_string(data: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    """Read an optional non-empty string field from one of several aliases."""

    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def required_string_list(data: Mapping[str, Any], keys: tuple[str, ...]) -> list[str]:
    """Read a required non-empty string list field from one of several aliases."""

    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            contexts = [item.strip() for item in value if isinstance(item, str) and item.strip()]
            if contexts:
                return contexts
    raise EvaluationDataError(f"Missing required string list field '{keys[0]}'.")


def require_examples(examples: list[EvaluationExample]) -> list[EvaluationExample]:
    """Require at least one evaluation example."""

    if not examples:
        raise EvaluationDataError("Evaluation dataset must contain at least one example.")
    return examples


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Run optional RAGAS evaluation for DocRAG.")
    parser.add_argument("dataset", type=Path, help="Path to .json or .jsonl evaluation data.")
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated RAGAS metric names.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Load and shape-check the dataset and metric names without running RAGAS "
            "(no ragas/datasets import, no LLM calls) — a CI-safe smoke check that "
            "the shipped dataset stays loadable, independent of the optional eval extra."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.validate_only:
        try:
            examples = load_examples(args.dataset)
            metric_names = parse_metric_names(args.metrics)
        except EvaluationDataError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"validated {len(examples)} examples, {len(metric_names)} metrics")
        return 0

    try:
        examples = load_examples(args.dataset)
        metric_names = parse_metric_names(args.metrics)
        result = run_ragas_evaluation(examples, metric_names=metric_names)
        payload = result_to_jsonable(result)
    except (EvaluationDataError, RagasUnavailableError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(f"{output}\n", encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

