import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MLflow + scikit-learn: tracking, logging, and tuning

    This notebook walks through three things, each in its own section:

    1. **Load** the classic Wine dataset.
    2. **Train + log** a single scikit-learn model with MLflow (the "one run" pattern).
    3. **Tune** that model with Optuna, logging every trial as its own MLflow run, then
       picking and re-logging the best one.

    The core idea behind MLflow here: every time you train something, wrap it in
    `mlflow.start_run()` and log three things — the **parameters** that produced the
    model, the **metrics** that resulted, and the **model artifact** itself. That turns
    "I ran some experiments" into a queryable, comparable history.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 0 — Setup

    Two choices matter here:

    - **Tracking URI**: where MLflow stores run data. The default is a local
      `mlruns/` folder using flat files. We instead point it at a local **SQLite**
      database (`sqlite:///mlflow.db`). This costs nothing extra locally, but it
      unlocks the **Model Registry** (`registered_model_name=...`), which the flat-file
      store doesn't support. If you don't need the registry, `mlruns/` is fine too.
    - **Experiment**: a named container for runs. Grouping runs under one experiment
      (`wine-classification`) is what lets us later query "all runs from this
      experiment" and compare them.
    """)
    return


@app.cell
def _():
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("wine-classification")

    client = MlflowClient()
    return client, mlflow


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 1 — Load the Wine dataset

    The [Wine dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-dataset)
    is a small, classic multiclass classification problem: 178 samples, 13 numeric
    features from a chemical analysis of wines, and 3 target classes (cultivars).
    It's a good "boring" dataset for an MLflow example, since we don't have to spend
    any time on data cleaning and can focus entirely on the tracking mechanics.
    """)
    return


@app.cell
def _():
    from sklearn.datasets import load_wine
    import pandas as pd

    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name="target")

    dataset_summary = pd.DataFrame(
        {
            "n_samples": [X.shape[0]],
            "n_features": [X.shape[1]],
            "n_classes": [y.nunique()],
        }
    )
    return X, dataset_summary, pd, wine, y


@app.cell
def _(X):
    X
    return


@app.cell
def _(dataset_summary):
    dataset_summary
    return


@app.cell
def _(wine):

    wine.target_names
    return


@app.cell
def _(X, y):
    from sklearn.model_selection import train_test_split

    # stratify=y keeps the class proportions equal in train/test, which matters
    # because this dataset's 3 classes aren't perfectly balanced.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
    return X_test, X_train, X_val, y_test, y_train, y_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 2 — Basic training + logging with MLflow

    The pattern below is the same for almost any scikit-learn model:

    1. Open a run with `mlflow.start_run()`.
    2. `mlflow.log_params(...)` — log the hyperparameters *before* training, so a run
       is fully described even if training later fails.
    3. Fit the model as usual.
    4. `mlflow.log_metrics(...)` — log whatever you compute on the held-out set.
    5. `mlflow.sklearn.log_model(...)` — log the fitted estimator itself as an
       artifact. This is what lets you reload the exact model later with
       `mlflow.sklearn.load_model(...)`, without re-running any training code.

    We'll use a `DecisionTreeClassifier` here since it's simple and its
    hyperparameters (`max_depth`, `criterion`, ...) are easy to reason about — good
    for a first example before we hand tuning off to Optuna in Section 3.
    """)
    return


@app.cell
def _(X_train, X_val, mlflow, y_train, y_val):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    with mlflow.start_run(run_name="decision_tree_baseline") as baseline_run:
        baseline_params = {"max_depth": 4, "criterion": "gini", "random_state": 42}
        mlflow.log_params(baseline_params)

        baseline_clf = DecisionTreeClassifier(**baseline_params)
        baseline_clf.fit(X_train, y_train)
        baseline_preds = baseline_clf.predict(X_val)

        baseline_metrics = {
            "accuracy": accuracy_score(y_val, baseline_preds),
            "f1_macro": f1_score(y_val, baseline_preds, average="macro"),
            "precision_macro": precision_score(y_val, baseline_preds, average="macro"),
            "recall_macro": recall_score(y_val, baseline_preds, average="macro"),
        }
        mlflow.log_metrics(baseline_metrics)

        # Logging the model itself as an artifact of the run. input_example is
        # optional but nice: MLflow uses it to record the model's expected schema.
        mlflow.sklearn.log_model(
            baseline_clf, name="model", input_example=X_train.iloc[:5]
        )

        baseline_run_id = baseline_run.info.run_id

    baseline_metrics
    return (
        DecisionTreeClassifier,
        accuracy_score,
        baseline_metrics,
        baseline_run_id,
        f1_score,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That run is now durably recorded. Instead of trusting the Python variable
    `baseline_metrics` still sitting in memory, let's pull it back **from MLflow**
    using the run id — this is what "logged" buys you: the run survives even if
    you restart this notebook.
    """)
    return


@app.cell
def _(baseline_run_id, client):
    logged_run = client.get_run(baseline_run_id)

    logged_params = logged_run.data.params
    logged_metrics = logged_run.data.metrics
    logged_artifacts = [a.path for a in client.list_artifacts(baseline_run_id)]

    logged_params, logged_metrics, logged_artifacts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `logged_artifacts` should show a `model` folder — that's the serialized
    estimator plus its metadata (conda/pip requirements, input signature, etc.),
    everything needed to reload it elsewhere with:

    ```python
    mlflow.sklearn.load_model(f"runs:/{baseline_run_id}/model")
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Section 3 — Hyperparameter tuning with Optuna + MLflow

    The idea: let Optuna propose hyperparameter combinations, and log **every trial
    as its own nested MLflow run** under one parent run. Nesting matters for
    organization — later, you can filter "give me all trials that belong to this
    specific tuning session" using `tags.mlflow.parentRunId`, instead of every trial
    polluting the top-level run list.

    The objective function does the same 5 steps as Section 2 (log params → fit →
    log metrics), just wrapped in a function Optuna calls repeatedly, and it
    `return`s the metric Optuna should optimize.
    """)
    return


@app.cell
def _(
    DecisionTreeClassifier,
    X_train,
    X_val,
    f1_score,
    mlflow,
    y_train,
    y_val,
):
    import optuna

    # Quiet Optuna's per-trial logging so it doesn't spam notebook output;
    # we'll inspect results via MLflow instead.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        trial_params = {
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        }

        # nested=True is what attaches this run to the currently active
        # (parent) run instead of starting an unrelated top-level run.
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            mlflow.log_params(trial_params)

            trial_clf = DecisionTreeClassifier(**trial_params, random_state=42)
            trial_clf.fit(X_train, y_train)
            trial_preds = trial_clf.predict(X_val)

            f1 = f1_score(y_val, trial_preds, average="macro")
            mlflow.log_metric("f1_macro", f1)

        return f1

    return objective, optuna


@app.cell
def _(mlflow, objective, optuna):
    N_TRIALS = 30

    with mlflow.start_run(run_name="optuna_tuning_parent") as parent_run:
        study = optuna.create_study(
            direction="maximize", study_name="wine_dt_tuning"
        )
        study.optimize(objective, n_trials=N_TRIALS)

        # Log the study-level summary on the parent run too, so the parent
        # run itself is a self-contained record of "what search did we run,
        # and what did it find".
        mlflow.log_params({"n_trials": N_TRIALS, "direction": "maximize"})
        mlflow.log_metric("best_f1_macro", study.best_value)
        for _k, _v in study.best_params.items():
            mlflow.log_param(f"best_{_k}", _v)

        parent_run_id = parent_run.info.run_id

    study.best_value, study.best_params
    return parent_run_id, study


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparing all trials

    Rather than trusting the in-memory `study` object, we query MLflow for every run
    whose parent is our tuning session, and sort by the metric we optimized. This is
    the same query you'd run days later, in a different process, to see how a past
    tuning session went.
    """)
    return


@app.cell
def _(mlflow, parent_run_id):
    trials_df = mlflow.search_runs(
        experiment_names=["wine-classification"],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        order_by=["metrics.f1_macro DESC"],
    )

    trials_view = trials_df[
        [
            "run_id",
            "params.max_depth",
            "params.min_samples_split",
            "params.min_samples_leaf",
            "params.criterion",
            "metrics.f1_macro",
        ]
    ].reset_index(drop=True)

    trials_view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Picking the best model

    `study.best_params` already gives us the winning hyperparameters directly from
    Optuna (no need to re-parse the MLflow table for this — Optuna tracked it as it
    went). We retrain once more on those params and log this as its own clearly
    named run, and this time we also **register** it in the MLflow Model Registry
    with `registered_model_name`. Registering assigns it a stable name +
    version (`wine-classifier-best`, version 1, 2, 3...) you can refer to later
    without needing to remember a run id.
    """)
    return


@app.cell
def _(study):
    best_params = study.best_params
    best_params
    return (best_params,)


@app.cell
def _(
    DecisionTreeClassifier,
    X_train,
    X_val,
    accuracy_score,
    best_params,
    f1_score,
    mlflow,
    y_train,
    y_val,
):
    with mlflow.start_run(run_name="best_model_final") as best_run:
        best_clf = DecisionTreeClassifier(**best_params, random_state=42)
        best_clf.fit(X_train, y_train)
        best_preds = best_clf.predict(X_val)

        best_metrics = {
            "accuracy": accuracy_score(y_val, best_preds),
            "f1_macro": f1_score(y_val, best_preds, average="macro"),
        }
        mlflow.log_params(best_params)
        mlflow.log_metrics(best_metrics)
        mlflow.sklearn.log_model(
            best_clf,
            name="model",
            registered_model_name="wine-classifier-best",
        )

    best_params, best_metrics
    return best_clf, best_metrics


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Baseline vs. tuned

    Finally, a direct comparison of where we started (Section 2's fixed
    `max_depth=4` tree) versus where Optuna's search landed.
    """)
    return


@app.cell
def _(baseline_metrics, best_metrics, pd):
    comparison = pd.DataFrame(
        {
            "baseline (fixed params)": baseline_metrics,
            "tuned (optuna best)": best_metrics,
        }
    ).T

    comparison
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Use the test set to verify classification with held out set.

    >**Important**: Here for the first time we use the test set that has been held out for al the training process.
    """)
    return


@app.cell
def _(X_test, accuracy_score, best_clf, f1_score, y_test):
    test_preds = best_clf.predict(X_test)

    test_metrics = {
        "accuracy": accuracy_score(y_test, test_preds),
        "f1_macro": f1_score(y_test, test_preds, average="macro"),
    }
    test_metrics
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    **To view all of this visually**, run, from a terminal in the same directory as
    `mlflow.db`:

    ```bash
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    ```

    and open the printed `localhost` URL. You'll see the `wine-classification`
    experiment with the baseline run, the `optuna_tuning_parent` run with all 30
    trials nested underneath it, and `best_model_final` — each with its logged
    params, metrics, and downloadable model artifact.
    """)
    return


if __name__ == "__main__":
    app.run()
