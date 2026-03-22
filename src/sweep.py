"""
Bayesian hyperparameter sweep with Optuna + MLflow tracking.

Uses Optuna's Tree-structured Parzen Estimator (TPE) to "intelligently"
explore hyperparameter spaces instead of brute-force grid search.
Also includes a Soft-Voting Classifier built from the best individual models.
"""

import warnings

import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# silence Optuna's verbose trial logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Model registry
# Each entry maps a model name to a function that receives an Optuna trial
# and returns (model_instance, param_dict_for_logging).

MODEL_REGISTRY: dict[str, str] = {}  # filled by decorator
_SUGGEST_FNS: dict[str, callable] = {}


def _register(name: str):
    """Decorator to register an Optuna suggest function for a model."""

    def decorator(fn):
        _SUGGEST_FNS[name] = fn
        MODEL_REGISTRY[name] = name
        return fn

    return decorator


@_register("RandomForest")
def _suggest_rf(trial: optuna.Trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_categorical("max_depth", [10, 20, 30, None]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
    }
    return RandomForestClassifier(**params, n_jobs=-1, random_state=42), params


@_register("SVM")
def _suggest_svm(trial: optuna.Trial):
    kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])
    params = {
        "C": trial.suggest_float("C", 0.01, 100, log=True),
        "kernel": kernel,
    }
    if kernel in ("rbf", "poly"):
        params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
    if kernel == "poly":
        params["degree"] = trial.suggest_int("degree", 2, 4)
    # probability=True is needed for soft voting later
    return SVC(**params, probability=True, random_state=42, class_weight="balanced"), params


@_register("GradientBoosting")
def _suggest_gb(trial: optuna.Trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    }
    return GradientBoostingClassifier(**params, random_state=42), params


@_register("KNN")
def _suggest_knn(trial: optuna.Trial):
    params = {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 25),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
    }
    return KNeighborsClassifier(**params, n_jobs=-1), params


@_register("LogisticRegression")
def _suggest_lr(trial: optuna.Trial):
    params = {
        "C": trial.suggest_float("C", 0.001, 100, log=True),
        "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
        "max_iter": trial.suggest_int("max_iter", 300, 1500, step=100),
    }
    return LogisticRegression(**params, random_state=42, n_jobs=-1), params


# Evaluation
def evaluate_model(model, X_test, y_test) -> dict:
    """Return a dict of common classification metrics."""
    y_pred = model.predict(X_test)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "precision_weighted": precision_score(y_test, y_pred, average="weighted"),
            "recall_weighted": recall_score(y_test, y_pred, average="weighted"),
        }


# Bayesian sweep
def run_sweep(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_names: list[str] | None = None,
    experiment_name: str = "SER_bayesian_sweep",
    n_trials_per_model: int = 30,
) -> pd.DataFrame:
    """
    Bayesian hyperparameter sweep (Optuna TPE) over selected models,
    with every trial logged to MLflow.

    Parameters
    ----------
    model_names : list of keys in MODEL_REGISTRY, or None for all.
    experiment_name : MLflow experiment name.
    n_trials_per_model : number of Optuna trials per model family.

    Returns
    -------
    DataFrame with one row per trial (params + metrics + run_id).
    """
    mlflow.set_experiment(experiment_name)

    if model_names is None:
        model_names = list(MODEL_REGISTRY.keys())

    results = []

    for name in model_names:
        suggest_fn = _SUGGEST_FNS[name]

        print(f"\n{'=' * 60}")
        print(f"  {name} — {n_trials_per_model} Bayesian trials (TPE)")
        print(f"{'=' * 60}")

        def objective(trial, suggest_fn=suggest_fn, name=name):
            model, params = suggest_fn(trial)
            run_name = f"{name}_trial{trial.number}"

            with mlflow.start_run(run_name=run_name):
                mlflow.set_tag("model_type", name)
                mlflow.set_tag("search_method", "bayesian_tpe")
                mlflow.log_params(params)

                model.fit(X_train, y_train)
                metrics = evaluate_model(model, X_test, y_test)
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, artifact_path="model")

                print(
                    f"  [trial {trial.number:2d}] acc={metrics['accuracy']:.4f}  "
                    f"f1={metrics['f1_weighted']:.4f}  | {params}"
                )

                results.append(
                    {
                        "run_id": mlflow.active_run().info.run_id,
                        "model_type": name,
                        "run_name": run_name,
                        **params,
                        **metrics,
                    }
                )

            return metrics["f1_weighted"]

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            study_name=f"{experiment_name}_{name}",
        )
        study.optimize(objective, n_trials=n_trials_per_model, show_progress_bar=False)

        print(f"\n  [OK] {name} best F1={study.best_value:.4f}  params={study.best_params}")

    return pd.DataFrame(results)


# Voting Classifier
def build_voting_classifier(
    results_df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    top_n: int = 3,
    voting: str = "soft",
    experiment_name: str = "SER_bayesian_sweep",
) -> tuple:
    """
    Build a VotingClassifier from the top-N best individual models
    found during the sweep (one per model family).

    Parameters
    ----------
    results_df : sweep results DataFrame.
    top_n : number of best model families to include.
    voting : 'soft' (probability-weighted) or 'hard' (majority vote).

    Returns
    -------
    (voting_model, metrics_dict, run_id)
    """
    mlflow.set_experiment(experiment_name)

    # Get the best run per model family, then take top-N
    best_per_family = (
        results_df.sort_values("f1_weighted", ascending=False)
        .drop_duplicates(subset="model_type", keep="first")
        .head(top_n)
    )

    print(f"\n{'=' * 60}")
    print(f"  VotingClassifier ({voting}) — top {top_n} model families")
    print(f"{'=' * 60}")

    # Load each best model from MLflow
    estimators = []
    for _, row in best_per_family.iterrows():
        model = mlflow.sklearn.load_model(f"runs:/{row['run_id']}/model")
        estimators.append((row["model_type"], model))
        print(f"  • {row['model_type']:20s}  F1={row['f1_weighted']:.4f}")

    # Build and train the ensemble
    vc = VotingClassifier(estimators=estimators, voting=voting, n_jobs=-1)
    vc.fit(X_train, y_train)

    metrics = evaluate_model(vc, X_test, y_test)

    # Log to MLflow
    with mlflow.start_run(run_name=f"VotingClassifier_{voting}"):
        mlflow.set_tag("model_type", "VotingClassifier")
        mlflow.set_tag("voting", voting)
        mlflow.set_tag("estimators", ", ".join([e[0] for e in estimators]))
        mlflow.log_param("top_n", top_n)
        mlflow.log_param("voting", voting)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(vc, artifact_path="model")
        run_id = mlflow.active_run().info.run_id

    print(f"\n  VotingClassifier  acc={metrics['accuracy']:.4f}  f1={metrics['f1_weighted']:.4f}")

    return vc, metrics, run_id


# Helpers
def get_best_run(results_df: pd.DataFrame, metric: str = "f1_weighted") -> pd.Series:
    """Return the row with the highest value of `metric`."""
    return results_df.loc[results_df[metric].idxmax()]
