"""Evaluation metrics, checked against hand-computed examples."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from speechdsp.metrics import confusion_report, cross_val_report, sensitivity_specificity, uar

# Four negatives and two positives; one negative and one positive are wrong.
Y_TRUE = [0, 0, 0, 0, 1, 1]
Y_PRED = [0, 0, 0, 1, 1, 0]


def test_uar_matches_the_hand_computed_value():
    # recall(0) = 3/4, recall(1) = 1/2  ->  UAR = 0.625
    assert uar(Y_TRUE, Y_PRED) == pytest.approx(0.625)


def test_uar_of_a_perfect_classifier_is_one():
    assert uar([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)


def test_uar_of_a_majority_only_predictor_is_one_half():
    assert uar([0] * 9 + [1], [0] * 10) == pytest.approx(0.5)


def test_uar_is_not_fooled_by_class_imbalance():
    y_true = [0] * 95 + [1] * 5
    y_pred = [0] * 100
    assert uar(y_true, y_pred) == pytest.approx(0.5)


def test_uar_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        uar([0, 1], [0, 1, 1])


def test_sensitivity_specificity_matches_the_hand_computed_value():
    # TP = 1, FN = 1, TN = 3, FP = 1
    sens, spec = sensitivity_specificity(Y_TRUE, Y_PRED, pos_label=1)
    assert sens == pytest.approx(0.5)
    assert spec == pytest.approx(0.75)


def test_sensitivity_specificity_honours_a_custom_positive_label():
    sens, spec = sensitivity_specificity(Y_TRUE, Y_PRED, pos_label=0)
    assert sens == pytest.approx(0.75)
    assert spec == pytest.approx(0.5)


def test_sensitivity_specificity_works_with_string_labels():
    sens, spec = sensitivity_specificity(
        ["ill", "ill", "healthy"], ["ill", "healthy", "healthy"], pos_label="ill"
    )
    assert sens == pytest.approx(0.5)
    assert spec == pytest.approx(1.0)


def test_sensitivity_is_undefined_without_positive_samples():
    sens, spec = sensitivity_specificity([0, 0, 0], [0, 0, 1], pos_label=1)
    assert np.isnan(sens)
    assert spec == pytest.approx(2 / 3)


def test_confusion_report_contents():
    rep = confusion_report(Y_TRUE, Y_PRED)
    np.testing.assert_array_equal(rep["confusion_matrix"], [[3, 1], [1, 1]])
    np.testing.assert_array_equal(rep["support"], [4, 2])
    assert rep["labels"] == [0, 1]
    assert rep["accuracy"] == pytest.approx(4 / 6)
    assert rep["uar"] == pytest.approx(0.625)
    assert rep["sensitivity"] == pytest.approx(0.5)
    assert rep["specificity"] == pytest.approx(0.75)
    assert rep["per_class_recall"][0] == pytest.approx(0.75)
    assert rep["per_class_precision"][1] == pytest.approx(0.5)


def test_confusion_report_keeps_a_requested_label_order():
    rep = confusion_report([0, 1], [0, 1], labels=[1, 0])
    assert rep["labels"] == [1, 0]
    np.testing.assert_array_equal(rep["confusion_matrix"], [[1, 0], [0, 1]])


def test_confusion_report_leaves_sensitivity_undefined_for_three_classes():
    rep = confusion_report([0, 1, 2, 2], [0, 1, 2, 1])
    assert np.isnan(rep["sensitivity"])
    assert rep["uar"] == pytest.approx((1.0 + 1.0 + 0.5) / 3)


def test_confusion_report_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        confusion_report([], [])


def make_separable(n_per_class: int = 40, seed: int = 3):
    """Two well-separated Gaussian blobs."""
    gen = np.random.default_rng(seed)
    X = np.vstack(
        [
            gen.normal(-1.5, 0.5, (n_per_class, 4)),
            gen.normal(+1.5, 0.5, (n_per_class, 4)),
        ]
    )
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


def test_cross_val_report_structure_and_scores():
    X, y = make_separable()
    rep = cross_val_report(LogisticRegression(), X, y, n_splits=5, seed=0)
    assert rep["n_splits"] == 5
    assert len(rep["folds"]) == 5
    assert rep["labels"] == [0, 1]
    assert rep["pooled"]["uar"] > 0.9
    assert rep["mean"]["uar"] > 0.9
    assert rep["std"]["uar"] >= 0.0
    assert sum(f["n_test"] for f in rep["folds"]) == y.size
    for fold in rep["folds"]:
        assert set(fold) >= {"fold", "n_train", "n_test", "uar", "confusion_matrix"}


def test_cross_val_report_is_reproducible_for_a_fixed_seed():
    X, y = make_separable()
    first = cross_val_report(LogisticRegression(), X, y, n_splits=4, seed=42)
    second = cross_val_report(LogisticRegression(), X, y, n_splits=4, seed=42)
    assert first["pooled"]["uar"] == second["pooled"]["uar"]
    np.testing.assert_array_equal(
        first["pooled"]["confusion_matrix"], second["pooled"]["confusion_matrix"]
    )


def test_cross_val_report_does_not_mutate_the_estimator():
    X, y = make_separable()
    estimator = LogisticRegression()
    cross_val_report(estimator, X, y, n_splits=3)
    assert not hasattr(estimator, "coef_")


def test_cross_val_report_scores_a_majority_classifier_at_chance():
    X, y = make_separable()
    rep = cross_val_report(DummyClassifier(strategy="most_frequent"), X, y, n_splits=4)
    assert rep["pooled"]["uar"] == pytest.approx(0.5)


def test_cross_val_report_keeps_groups_inside_one_fold():
    X, y = make_separable(n_per_class=30)
    groups = np.concatenate([np.repeat(np.arange(6), 5), np.repeat(np.arange(6, 12), 5)])
    rep = cross_val_report(LogisticRegression(), X, y, groups=groups, n_splits=3)
    assert len(rep["folds"]) == 3
    assert rep["pooled"]["uar"] > 0.9


def test_cross_val_report_rejects_a_class_smaller_than_the_fold_count():
    X, y = make_separable(n_per_class=10)
    y = y.copy()
    y[:9] = 2  # leaves a class with a single member
    with pytest.raises(ValueError, match="rarest class"):
        cross_val_report(LogisticRegression(), X, y, n_splits=5)


def test_cross_val_report_rejects_mismatched_shapes():
    X, y = make_separable(n_per_class=10)
    with pytest.raises(ValueError, match="rows"):
        cross_val_report(LogisticRegression(), X, y[:-1], n_splits=3)
