from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from eval.ragas_runner import (
    EvaluationDataError,
    EvaluationExample,
    RagasUnavailableError,
    load_examples,
    load_ragas_runtime,
    parse_metric_names,
    result_to_jsonable,
    run_ragas_evaluation,
)


def test_evaluation_example_accepts_docrag_and_ragas_field_names() -> None:
    example = EvaluationExample.from_mapping(
        {
            "question": "What is DocRAG?",
            "answer": "A document Q&A app.",
            "contexts": ["DocRAG answers from uploaded docs."],
            "ground_truth": "DocRAG is a document Q&A app.",
        }
    )

    assert example.to_ragas_row() == {
        "user_input": "What is DocRAG?",
        "response": "A document Q&A app.",
        "retrieved_contexts": ["DocRAG answers from uploaded docs."],
        "reference": "DocRAG is a document Q&A app.",
    }

    ragas_named = EvaluationExample.from_mapping(
        {
            "user_input": "Question?",
            "response": "Answer.",
            "retrieved_contexts": ["Context."],
        }
    )
    assert ragas_named.reference is None


def test_load_examples_supports_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "examples.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "Q1",
                "answer": "A1",
                "contexts": ["C1"],
                "reference": "R1",
            }
        )
        + "\n\n",
        encoding="utf-8",
    )

    examples = load_examples(path)

    assert len(examples) == 1
    assert examples[0].question == "Q1"


def test_load_examples_supports_json_object_with_examples(tmp_path: Path) -> None:
    path = tmp_path / "examples.json"
    path.write_text(
        json.dumps({"examples": [{"question": "Q", "answer": "A", "contexts": ["C"]}]}),
        encoding="utf-8",
    )

    assert load_examples(path)[0].answer == "A"


def test_load_examples_rejects_malformed_rows(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps([{"question": "Q", "answer": "A", "contexts": []}]),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationDataError, match="contexts"):
        load_examples(path)


def test_run_ragas_evaluation_uses_ragas_row_shape_with_fakes() -> None:
    seen: dict[str, Any] = {}

    def dataset_factory(rows: list[dict[str, Any]]) -> dict[str, Any]:
        seen["rows"] = rows
        return {"dataset_rows": rows}

    def evaluate_func(**kwargs: Any) -> dict[str, Any]:
        seen["kwargs"] = kwargs
        return {"faithfulness": 1.0}

    result = run_ragas_evaluation(
        [
            EvaluationExample(
                question="Q",
                answer="A",
                contexts=["C"],
                reference="R",
            )
        ],
        evaluate_func=evaluate_func,
        dataset_factory=dataset_factory,
        metric_objects=["faithfulness"],
    )

    assert result == {"faithfulness": 1.0}
    assert seen["rows"] == [
        {
            "user_input": "Q",
            "response": "A",
            "retrieved_contexts": ["C"],
            "reference": "R",
        }
    ]
    assert seen["kwargs"]["dataset"] == {"dataset_rows": seen["rows"]}
    assert seen["kwargs"]["metrics"] == ["faithfulness"]
    assert seen["kwargs"]["raise_exceptions"] is False
    assert seen["kwargs"]["show_progress"] is False


def test_load_ragas_runtime_reports_optional_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> Any:
        if name == "ragas":
            raise ModuleNotFoundError("ragas unavailable")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(RagasUnavailableError, match="RAGAS evaluation dependencies"):
        load_ragas_runtime(("faithfulness",))


def test_parse_metric_names_requires_at_least_one_metric() -> None:
    assert parse_metric_names("faithfulness, answer_relevancy") == (
        "faithfulness",
        "answer_relevancy",
    )
    with pytest.raises(EvaluationDataError):
        parse_metric_names(" , ")


def test_result_to_jsonable_prefers_to_dict() -> None:
    class Result:
        def to_dict(self) -> dict[str, float]:
            return {"score": 0.5}

    assert result_to_jsonable(Result()) == {"score": 0.5}
