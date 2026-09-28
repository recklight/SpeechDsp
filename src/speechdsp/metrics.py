"""Classification metrics and cross-validation reporting for imbalanced corpora.

Pathological-voice and other clinical corpora are almost always imbalanced, so
plain accuracy is a misleading summary: a classifier that always predicts the
majority class can look excellent.  The unweighted average recall (UAR, the mean
of the per-class recalls) and the sensitivity/specificity pair are reported
instead, both here and by :func:`cross_val_report`.

References
----------
.. [1] A. Rosenberg, "Classifying skewed data: importance weighting to optimize
       average recall", *Interspeech*, 2012.
.. [2] T. Fawcett, "An introduction to ROC analysis", *Pattern Recognition
       Letters*, 27(8):861-874, 2006.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

__all__ = [
    "confusion_report",
    "cross_val_report",
    "sensitivity_specificity",
    "uar",
]

_LOGGER = logging.getLogger(__name__)


def _as_labels(y: np.ndarray) -> np.ndarray:
    """Flatten a label vector without forcing a dtype (labels may be strings)."""
    return np.asarray(y).ravel()


def _per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Recall of every label in ``labels``; a label with no support yields 0."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    support = cm.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(support > 0, np.diag(cm) / np.maximum(support, 1), 0.0)
    return recall.astype(np.float64)


def uar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unweighted average recall: the mean of the per-class recalls.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground-truth labels.
    y_pred : numpy.ndarray
        Predicted labels, same length as ``y_true``.

    Returns
    -------
    float
        Mean recall over the classes present in ``y_true`` or ``y_pred``.  Also
        known as balanced accuracy; equals accuracy when the classes are
        balanced, and 0.5 for a two-class majority-only predictor.

    Examples
    --------
    >>> uar([0, 0, 0, 1], [0, 0, 0, 0])
    0.5

    References
    ----------
    .. [1] A. Rosenberg, "Classifying skewed data: importance weighting to
           optimize average recall", *Interspeech*, 2012.
    """
    y_true, y_pred = _as_labels(y_true), _as_labels(y_pred)
    if y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must have the same length")
    if y_true.size == 0:
        raise ValueError("cannot compute UAR from empty label vectors")
    labels = np.unique(np.concatenate([y_true, y_pred]))
    return float(np.mean(_per_class_recall(y_true, y_pred, labels)))


def sensitivity_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: Any = 1,
) -> tuple[float, float]:
    """Sensitivity (recall of the positive class) and specificity.

    Every label different from ``pos_label`` counts as negative, so the function
    also works on multi-class labels in a one-vs-rest sense.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground-truth labels.
    y_pred : numpy.ndarray
        Predicted labels.
    pos_label : Any, optional
        Label treated as positive, default ``1``.

    Returns
    -------
    sensitivity : float
        ``TP / (TP + FN)``, or ``nan`` if there is no positive sample.
    specificity : float
        ``TN / (TN + FP)``, or ``nan`` if there is no negative sample.

    Examples
    --------
    >>> sensitivity_specificity([1, 1, 0, 0], [1, 0, 0, 0])
    (0.5, 1.0)
    """
    y_true, y_pred = _as_labels(y_true), _as_labels(y_pred)
    if y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must have the same length")
    true_pos_mask = y_true == pos_label
    pred_pos_mask = y_pred == pos_label

    n_pos = int(np.count_nonzero(true_pos_mask))
    n_neg = int(true_pos_mask.size - n_pos)
    tp = int(np.count_nonzero(true_pos_mask & pred_pos_mask))
    tn = int(np.count_nonzero(~true_pos_mask & ~pred_pos_mask))

    sensitivity = tp / n_pos if n_pos else float("nan")
    specificity = tn / n_neg if n_neg else float("nan")
    if not n_pos or not n_neg:
        _LOGGER.warning("only one class is present; sensitivity/specificity is partly undefined")
    return float(sensitivity), float(specificity)


def confusion_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[Any] | None = None,
) -> dict[str, Any]:
    """Summarise a set of predictions as a dictionary of metrics.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground-truth labels.
    y_pred : numpy.ndarray
        Predicted labels.
    labels : list, optional
        Label order for the confusion matrix.  ``None`` (default) uses the
        sorted union of the labels that occur.

    Returns
    -------
    dict
        Keys: ``labels`` (list), ``confusion_matrix`` (``(n, n)`` int array with
        true classes on the rows), ``support`` (per-class counts), ``accuracy``,
        ``uar``, ``macro_f1``, ``per_class_recall`` and ``per_class_precision``
        (label -> float), plus ``sensitivity`` and ``specificity`` which are
        filled in only for two-class problems (``nan`` otherwise).

    Examples
    --------
    >>> rep = confusion_report([0, 1, 1], [0, 1, 0])
    >>> rep["accuracy"], round(rep["uar"], 3)
    (0.6666666666666666, 0.75)
    """
    y_true, y_pred = _as_labels(y_true), _as_labels(y_pred)
    if y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must have the same length")
    if y_true.size == 0:
        raise ValueError("cannot build a confusion report from empty label vectors")

    label_arr = (
        np.unique(np.concatenate([y_true, y_pred])) if labels is None else np.asarray(labels)
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_arr)
    recall = _per_class_recall(y_true, y_pred, label_arr)
    precision = precision_score(y_true, y_pred, labels=label_arr, average=None, zero_division=0)

    if label_arr.size == 2:
        sens, spec = sensitivity_specificity(y_true, y_pred, pos_label=label_arr[1])
    else:
        sens, spec = float("nan"), float("nan")

    keys = [item.item() if hasattr(item, "item") else item for item in label_arr]
    return {
        "labels": keys,
        "confusion_matrix": cm,
        "support": cm.sum(axis=1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "uar": float(np.mean(recall)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=label_arr, average="macro", zero_division=0)
        ),
        "per_class_recall": dict(zip(keys, recall.tolist(), strict=True)),
        "per_class_precision": dict(zip(keys, np.asarray(precision).tolist(), strict=True)),
        "sensitivity": sens,
        "specificity": spec,
    }


def _nan_safe(values: list[float], reducer: Any) -> float:
    """Apply ``reducer`` to the finite entries, returning ``nan`` if there are none.

    ``numpy.nanmean`` emits a RuntimeWarning for an all-NaN slice, which happens
    routinely here: sensitivity and specificity are undefined for multi-class
    problems, so those columns are NaN in every fold.
    """
    finite = [v for v in values if np.isfinite(v)]
    return float(reducer(finite)) if finite else float("nan")


def _splitter(
    groups: np.ndarray | None,
    n_splits: int,
    seed: int,
) -> StratifiedKFold | StratifiedGroupKFold:
    """Stratified splitter, group-aware when ``groups`` is given.

    Grouping matters whenever several recordings come from the same speaker: if
    they were allowed to straddle a fold boundary the model could recognise the
    speaker instead of the condition, and the score would be optimistic.
    """
    if groups is None:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def cross_val_report(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Run stratified cross-validation and report per-fold and pooled metrics.

    A fresh clone of ``estimator`` is fitted on every training split, so the
    object passed in is never modified.  The split is seeded, which makes the
    whole report reproducible.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        Any estimator implementing ``fit`` and ``predict``.
    X : numpy.ndarray
        Design matrix of shape ``(n_samples, n_features)``.
    y : numpy.ndarray
        Labels of shape ``(n_samples,)``.
    groups : numpy.ndarray or None, optional
        Group identifiers (e.g. speaker IDs).  When given,
        :class:`sklearn.model_selection.StratifiedGroupKFold` keeps all samples
        of a group inside the same fold.
    n_splits : int, optional
        Number of folds, default 5.
    seed : int, optional
        Random seed for the shuffling, default 0.

    Returns
    -------
    dict
        Keys:

        ``n_splits``, ``labels``
            Configuration echo.
        ``folds``
            List of per-fold dictionaries as returned by
            :func:`confusion_report`, each with an extra ``fold`` index and the
            train/test sizes.
        ``mean``, ``std``
            Mean and population standard deviation across folds of ``uar``,
            ``accuracy``, ``macro_f1``, ``sensitivity`` and ``specificity``.
        ``pooled``
            :func:`confusion_report` of the concatenated out-of-fold
            predictions, i.e. every sample scored exactly once.

    Raises
    ------
    ValueError
        If the sizes do not match or a class has fewer members than ``n_splits``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> rng = np.random.default_rng(0)
    >>> X = np.vstack([rng.normal(-1, 0.5, (30, 2)), rng.normal(1, 0.5, (30, 2))])
    >>> y = np.array([0] * 30 + [1] * 30)
    >>> report = cross_val_report(LogisticRegression(), X, y, n_splits=3)
    >>> report["pooled"]["uar"] > 0.9
    True
    """
    X = np.asarray(X)
    y = _as_labels(y)
    if X.shape[0] != y.size:
        raise ValueError(f"X has {X.shape[0]} rows but y has {y.size} labels")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    counts = np.unique(y, return_counts=True)[1]
    if counts.min() < n_splits:
        raise ValueError(
            f"the rarest class has {counts.min()} samples, which is fewer than n_splits={n_splits}"
        )
    groups_arr = None if groups is None else _as_labels(groups)

    splitter = _splitter(groups_arr, n_splits, seed)
    labels = np.unique(y)
    folds: list[dict[str, Any]] = []
    oof_true: list[np.ndarray] = []
    oof_pred: list[np.ndarray] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups_arr)):
        model = clone(estimator)
        model.fit(X[train_idx], y[train_idx])
        pred = _as_labels(model.predict(X[test_idx]))
        truth = y[test_idx]
        oof_true.append(truth)
        oof_pred.append(pred)

        report = confusion_report(truth, pred, labels=labels)
        report["fold"] = fold
        report["n_train"] = int(train_idx.size)
        report["n_test"] = int(test_idx.size)
        folds.append(report)
        _LOGGER.debug("fold %d: uar=%.4f acc=%.4f", fold, report["uar"], report["accuracy"])

    tracked = ("uar", "accuracy", "macro_f1", "sensitivity", "specificity")
    mean = {k: _nan_safe([f[k] for f in folds], np.mean) for k in tracked}
    std = {k: _nan_safe([f[k] for f in folds], np.std) for k in tracked}
    pooled = confusion_report(np.concatenate(oof_true), np.concatenate(oof_pred), labels=labels)

    return {
        "n_splits": int(n_splits),
        "labels": pooled["labels"],
        "folds": folds,
        "mean": mean,
        "std": std,
        "pooled": pooled,
    }
