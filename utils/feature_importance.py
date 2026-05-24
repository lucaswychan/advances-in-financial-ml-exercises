from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.cv import PurgedKFold, cross_val_scores, _score_model


def _get_y_and_sample_weight(X_df, y_df):
    y = y_df["bin"] if isinstance(y_df, pd.DataFrame) else y_df
    sample_weight = (
        y_df["w"]
        if isinstance(y_df, pd.DataFrame) and "w" in y_df.columns
        else pd.Series(1.0, index=X_df.index)
    )
    return y, sample_weight


def mean_decrease_impurity(model, X_df, y_df, cv_n_splits=10):
    y, sample_weight = _get_y_and_sample_weight(X_df, y_df)
    fit = model.fit(X_df, y, sample_weight=sample_weight.values)

    if hasattr(fit, "estimators_"):
        importance = pd.DataFrame(
            {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)},
            index=X_df.columns,
        ).T
        importance.replace(0, np.nan, inplace=True)
        importance = pd.concat(
            {
                "mean": importance.mean(),
                "std": importance.std() * importance.shape[0] ** -0.5,
            },
            axis=1,
        )
    else:
        importance = pd.DataFrame(
            {"mean": fit.feature_importances_, "std": 0.0},
            index=X_df.columns,
        )

    importance["mean"] /= importance["mean"].sum()
    return importance


def mean_decrease_accuracy(
    model,
    X_df,
    y_df,
    cv_n_splits=10,
    pct_embargo=0.0,
    scoring_metric="neg_log_loss",
):
    if scoring_metric not in ["neg_log_loss", "accuracy"]:
        raise ValueError("scoring_metric must be 'neg_log_loss' or 'accuracy'.")

    y, sample_weight = _get_y_and_sample_weight(X_df, y_df)
    cv_gen = PurgedKFold(
        n_splits=cv_n_splits,
        t1=y_df["t1"],
        pct_embargo=pct_embargo,
    )
    base_scores = pd.Series(dtype=float)
    permuted_scores = pd.DataFrame(columns=X_df.columns, dtype=float)

    for i, (train, test) in enumerate(cv_gen.split(X=X_df)):
        X_train = X_df.iloc[train, :]
        y_train = y.iloc[train]
        w_train = sample_weight.iloc[train]
        X_test = X_df.iloc[test, :]
        y_test = y.iloc[test]
        w_test = sample_weight.iloc[test]

        fit = model.fit(X_train, y_train, sample_weight=w_train.values)
        base_scores.loc[i] = _score_model(
            fit,
            X_test,
            y_test,
            sample_weight=w_test.values,
            scoring_metric=scoring_metric,
        )

        for feature in X_df.columns:
            X_test_permuted = X_test.copy(deep=True)
            X_test_permuted[feature] = np.random.permutation(X_test_permuted[feature])
            permuted_scores.loc[i, feature] = _score_model(
                fit,
                X_test_permuted,
                y_test,
                sample_weight=w_test.values,
                scoring_metric=scoring_metric,
            )

    importance = (-permuted_scores).add(base_scores, axis=0)
    if scoring_metric == "neg_log_loss":
        importance = importance / -permuted_scores
    else:
        importance = importance / (1.0 - permuted_scores)

    return pd.concat(
        {
            "mean": importance.mean(),
            "std": importance.std() * importance.shape[0] ** -0.5,
        },
        axis=1,
    )


def single_feature_importance(
    model,
    X_df,
    y_df,
    cv_n_splits=10,
    pct_embargo=0.0,
    scoring_metric="neg_log_loss",
):
    if scoring_metric not in ["neg_log_loss", "accuracy"]:
        raise ValueError("scoring_metric must be 'neg_log_loss' or 'accuracy'.")

    y, sample_weight = _get_y_and_sample_weight(X_df, y_df)
    cv_gen = PurgedKFold(
        n_splits=cv_n_splits,
        t1=y_df["t1"],
        pct_embargo=pct_embargo,
    )
    importance = pd.DataFrame(columns=["mean", "std"], index=X_df.columns)

    for feature in X_df.columns:
        feature_scores = cross_val_scores(
            model,
            X_df[[feature]],
            y,
            cv_gen,
            sample_weight=sample_weight,
            scoring_metric=scoring_metric,
        )
        importance.loc[feature, "mean"] = feature_scores.mean()
        importance.loc[feature, "std"] = feature_scores.std() * feature_scores.shape[0] ** -0.5

    return importance.astype(float)


def plot_feature_importance(
    importance,
    oob_score=None,
    oos_score=None,
    method=None,
    tag=0,
    sim_num=0,
    savefig=False,
    output_path=None,
):
    plt.figure(figsize=(16, 10))
    importance = importance.sort_values("mean", ascending=True).copy()

    if method == "MDI":
        importance = importance / importance["mean"].sum()

    ax = importance["mean"].plot(
        kind="barh",
        color="b",
        alpha=0.25,
        xerr=importance["std"],
        error_kw={"ecolor": "r"},
    )

    if method == "MDI":
        ax.set_xlim([0, importance.sum(axis=1).max()])
        ax.axvline(1.0 / importance.shape[0], linewidth=1, color="r", linestyle="dotted")

    ax.get_yaxis().set_visible(False)
    for patch, feature_name in zip(ax.patches, importance.index):
        ax.text(
            patch.get_width() / 2,
            patch.get_y() + patch.get_height() / 2,
            feature_name,
            ha="center",
            va="center",
            color="black",
        )

    title_parts = [f"tag={tag}", f"sim_num={sim_num}"]
    if oob_score is not None:
        title_parts.append(f"oob={round(oob_score, 4)}")
    if oos_score is not None:
        title_parts.append(f"oos={round(oos_score, 4)}")
    plt.title(" | ".join(title_parts))

    if savefig:
        if output_path is None:
            output_path = f"feature_importance_{sim_num}.png"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100)

    plt.show()


def feature_importance(model, X_df, y_df, metric, cv_n_splits=10):
    if not isinstance(y_df, pd.DataFrame) or "t1" not in y_df.columns:
        raise ValueError("y_df must be a DataFrame with at least 'bin' and 't1' columns.")

    y, sample_weight = _get_y_and_sample_weight(X_df, y_df)
    fit = model.fit(X_df, y, sample_weight=sample_weight.values)
    oob_score = fit.oob_score_ if hasattr(fit, 'oob_score_') else None

    cv_gen = PurgedKFold(n_splits=cv_n_splits, t1=y_df['t1'])
    oos_score = cross_val_scores(
        model,
        X_df,
        y,
        cv_gen,
        sample_weight=sample_weight,
        scoring_metric="accuracy",
    ).mean()

    if metric == 'MDI':
        results = mean_decrease_impurity(model, X_df, y_df, cv_n_splits=cv_n_splits)
    elif metric == 'MDA':
        results = mean_decrease_accuracy(model, X_df, y_df, cv_n_splits=cv_n_splits, scoring_metric="accuracy")
    elif metric == 'SFI':
        results = single_feature_importance(model, X_df, y_df, cv_n_splits=cv_n_splits, scoring_metric="accuracy")
    else:
        raise ValueError("metric must be 'MDI', 'MDA', or 'SFI'.")
    
    return results, oob_score, oos_score


def run_test(model, X_df, y_df, cv_n_splits=10, run='', metrics=['MDI', 'MDA', 'SFI']):
    for metric in metrics:
        feature_imp, oob_score, oos_score = feature_importance(model, X_df, y_df, metric, cv_n_splits=cv_n_splits)

        plot_feature_importance(
            feature_imp, oob_score=oob_score, oos_score=oos_score,
            savefig=True, output_path='img/{}_feat_imp{}.png'.format(metric, run)
        )