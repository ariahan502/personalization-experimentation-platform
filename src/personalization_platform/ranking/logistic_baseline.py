from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


DEFAULT_NUMERIC_FEATURES = [
    "merged_rank",
    "normalized_merged_rank",
    "merged_score",
    "source_rank",
    "source_count",
    "candidate_count",
    "history_length",
    "history_click_count",
    "topic_history_count",
    "request_index_in_session",
    "request_hour",
]
DEFAULT_BINARY_FEATURES = [
    "has_multi_source_provenance",
    "has_affinity_source",
    "has_trending_source",
    "is_affinity_primary",
    "is_trending_primary",
    "is_cold_start",
    "candidate_seen_in_impressions",
]
DEFAULT_CATEGORICAL_FEATURES = [
    "candidate_source",
    "topic",
]


MODEL_NAME_BY_TYPE = {
    "logistic_regression": "logistic_regression_baseline",
    "random_forest": "random_forest_baseline",
}


def train_ranker_model(config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    dataset_dir = resolve_ranking_dataset_dir(config)
    dataset = pd.read_csv(dataset_dir / "ranking_dataset.csv")

    train_rows = dataset.loc[dataset["dataset_split"] == "train"].copy()
    valid_rows = dataset.loc[dataset["dataset_split"] == "valid"].copy()
    if train_rows.empty or valid_rows.empty:
        raise ValueError("Ranking dataset must contain both train and valid rows for smoke training.")

    feature_config = config.get("features", {})
    numeric_features = feature_config.get("numeric", DEFAULT_NUMERIC_FEATURES)
    binary_features = feature_config.get("binary", DEFAULT_BINARY_FEATURES)
    categorical_features = feature_config.get("categorical", DEFAULT_CATEGORICAL_FEATURES)

    vectorizer = DictVectorizer(sparse=True)
    train_dicts = build_feature_dicts(
        train_rows,
        numeric_features=numeric_features,
        binary_features=binary_features,
        categorical_features=categorical_features,
    )
    valid_dicts = build_feature_dicts(
        valid_rows,
        numeric_features=numeric_features,
        binary_features=binary_features,
        categorical_features=categorical_features,
    )

    x_train = vectorizer.fit_transform(train_dicts)
    x_valid = vectorizer.transform(valid_dicts)
    y_train = train_rows["label"].astype(int).to_numpy()
    y_valid = valid_rows["label"].astype(int).to_numpy()

    model_config = config.get("model", {})
    model_type = str(model_config.get("model_type", "logistic_regression"))
    model_name = resolve_model_name(model_type)
    model = build_model(model_type=model_type, model_config=model_config)
    x_train_input, x_valid_input = prepare_model_inputs(
        model_type=model_type,
        x_train=x_train,
        x_valid=x_valid,
    )
    model.fit(x_train_input, y_train)

    train_rows = train_rows.copy()
    valid_rows = valid_rows.copy()
    train_rows["prediction"] = model.predict_proba(x_train_input)[:, 1]
    valid_rows["prediction"] = model.predict_proba(x_valid_input)[:, 1]
    train_rows["predicted_label"] = model.predict(x_train_input)
    valid_rows["predicted_label"] = model.predict(x_valid_input)

    scored_rows = pd.concat([train_rows, valid_rows], ignore_index=True)
    scored_rows = scored_rows.sort_values(
        ["dataset_split", "request_ts", "request_id", "prediction"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)

    feature_names = vectorizer.get_feature_names_out().tolist()
    metrics = build_ranker_metrics(
        train_rows=train_rows,
        valid_rows=valid_rows,
        dataset_dir=dataset_dir,
        model_name=model_name,
    )
    manifest = build_ranker_manifest(
        config=config,
        metrics=metrics,
        model_name=model_name,
        model_type=model_type,
        numeric_features=numeric_features,
        binary_features=binary_features,
        categorical_features=categorical_features,
        feature_names=feature_names,
        model=model,
    )
    model_artifacts = {
        "vectorizer": vectorizer,
        "model": model,
        "model_type": model_type,
        "feature_names": feature_names,
    }
    return metrics | {"model_artifacts": model_artifacts}, scored_rows, manifest


def train_logistic_baseline(config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    logistic_config = dict(config)
    logistic_config["model"] = {"model_type": "logistic_regression"} | dict(config.get("model", {}))
    return train_ranker_model(logistic_config)


def resolve_model_name(model_type: str) -> str:
    if model_type not in MODEL_NAME_BY_TYPE:
        raise ValueError(
            f"Unsupported model_type={model_type!r}. Expected one of {sorted(MODEL_NAME_BY_TYPE)}."
        )
    return MODEL_NAME_BY_TYPE[model_type]


def build_model(*, model_type: str, model_config: dict[str, Any]) -> Any:
    if model_type == "logistic_regression":
        return LogisticRegression(
            max_iter=int(model_config.get("max_iter", 1000)),
            C=float(model_config.get("C", 1.0)),
            solver=model_config.get("solver", "liblinear"),
            random_state=int(model_config.get("random_state", 42)),
        )
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(model_config.get("n_estimators", 200)),
            max_depth=(
                int(model_config["max_depth"]) if model_config.get("max_depth") is not None else None
            ),
            min_samples_leaf=int(model_config.get("min_samples_leaf", 1)),
            random_state=int(model_config.get("random_state", 42)),
        )
    raise ValueError(f"Unsupported model_type={model_type!r}.")


def prepare_model_inputs(*, model_type: str, x_train: Any, x_valid: Any) -> tuple[Any, Any]:
    if model_type == "random_forest":
        return x_train.toarray(), x_valid.toarray()
    return x_train, x_valid


def resolve_ranking_dataset_dir(config: dict[str, Any]) -> Path:
    ranking_input = config["input"]
    base_dir = Path(ranking_input["ranking_dataset_base_dir"])
    run_name = ranking_input["ranking_dataset_run_name"]
    matches = sorted(base_dir.glob(f"*_{run_name}"))
    if not matches:
        raise FileNotFoundError(
            f"No ranking dataset outputs found under {base_dir} matching '*_{run_name}'."
        )
    return matches[-1]


def build_feature_dicts(
    rows: pd.DataFrame,
    *,
    numeric_features: list[str],
    binary_features: list[str],
    categorical_features: list[str],
) -> list[dict[str, Any]]:
    feature_rows: list[dict[str, Any]] = []
    for record in rows.to_dict(orient="records"):
        feature_row: dict[str, Any] = {}
        for column in numeric_features:
            feature_row[column] = float(record[column])
        for column in binary_features:
            feature_row[column] = int(record[column])
        for column in categorical_features:
            feature_row[column] = str(record[column])
        feature_rows.append(feature_row)
    return feature_rows


def build_ranker_metrics(
    *,
    train_rows: pd.DataFrame,
    valid_rows: pd.DataFrame,
    dataset_dir: Path,
    model_name: str,
) -> dict[str, Any]:
    train_metrics = build_split_metrics(train_rows)
    valid_metrics = build_split_metrics(valid_rows)
    valid_ranking_metrics = build_request_ranking_metrics(valid_rows)
    return {
        "model_name": model_name,
        "ranking_dataset_input_dir": str(dataset_dir),
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "valid_ranking_metrics": valid_ranking_metrics,
    }


def build_split_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    y_true = rows["label"].astype(int)
    y_pred = rows["predicted_label"].astype(int)
    y_score = rows["prediction"].astype(float)
    metrics = {
        "row_count": int(len(rows)),
        "positive_labels": int(y_true.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_score, labels=[0, 1])),
    }
    if y_true.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def build_request_ranking_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    reciprocal_ranks: list[float] = []
    hit_at_1: list[int] = []
    hit_at_3: list[int] = []

    for _, request_rows in rows.groupby("request_id"):
        ranked = request_rows.sort_values("prediction", ascending=False).reset_index(drop=True)
        positives = ranked.index[ranked["label"] == 1].tolist()
        if positives:
            best_rank = positives[0] + 1
            reciprocal_ranks.append(1.0 / best_rank)
            hit_at_1.append(int(best_rank <= 1))
            hit_at_3.append(int(best_rank <= 3))
        else:
            reciprocal_ranks.append(0.0)
            hit_at_1.append(0)
            hit_at_3.append(0)

    return {
        "request_count": int(rows["request_id"].nunique()),
        "mean_reciprocal_rank": float(sum(reciprocal_ranks) / len(reciprocal_ranks)),
        "hit_rate_at_1": float(sum(hit_at_1) / len(hit_at_1)),
        "hit_rate_at_3": float(sum(hit_at_3) / len(hit_at_3)),
    }


def build_ranker_manifest(
    *,
    config: dict[str, Any],
    metrics: dict[str, Any],
    model_name: str,
    model_type: str,
    numeric_features: list[str],
    binary_features: list[str],
    categorical_features: list[str],
    feature_names: list[str],
    model: Any,
) -> dict[str, Any]:
    manifest = {
        "model_name": model_name,
        "model_type": model_type,
        "ranking_dataset_input_dir": metrics["ranking_dataset_input_dir"],
        "feature_spec": {
            "numeric": numeric_features,
            "binary": binary_features,
            "categorical": categorical_features,
        },
        "feature_count_after_vectorization": len(feature_names),
        "assumptions": build_manifest_assumptions(model_type),
        "config_snapshot": config,
    }
    if model_type == "logistic_regression":
        coefficients = model.coef_[0].tolist()
        feature_weights = sorted(
            (
                {"feature": feature_name, "coefficient": float(coefficient)}
                for feature_name, coefficient in zip(feature_names, coefficients, strict=True)
            ),
            key=lambda row: abs(row["coefficient"]),
            reverse=True,
        )
        manifest["top_feature_weights"] = feature_weights[:10]
    elif hasattr(model, "feature_importances_"):
        feature_importances = sorted(
            (
                {"feature": feature_name, "importance": float(importance)}
                for feature_name, importance in zip(feature_names, model.feature_importances_, strict=True)
            ),
            key=lambda row: row["importance"],
            reverse=True,
        )
        manifest["top_feature_importances"] = feature_importances[:10]
    return manifest


def build_manifest_assumptions(model_type: str) -> list[str]:
    shared_assumptions = [
        "Smoke evaluation includes both classification metrics and request-level ranking metrics so later model comparisons have a shared baseline.",
        "The model is trained only on the generated ranking dataset contract and does not use hidden notebook-only transforms.",
    ]
    if model_type == "logistic_regression":
        return [
            "The first baseline ranker uses logistic regression for interpretability rather than maximum performance.",
            *shared_assumptions,
        ]
    if model_type == "random_forest":
        return [
            "The tree-based baseline trades some linear interpretability for non-linear feature interactions on the same ranking dataset contract.",
            *shared_assumptions,
        ]
    return shared_assumptions


def write_model_pickle(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(pickle.dumps(payload))


def sanitize_metrics_for_json(metrics: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(metrics)
    sanitized.pop("model_artifacts", None)
    return json.loads(json.dumps(sanitized, default=_json_default))


def _json_default(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable.")
