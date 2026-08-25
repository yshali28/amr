"""
Multi-label evaluation metrics.

Example:
    import numpy as np
    from src.evaluation.metrics import multilabel_metrics

    y_true = np.array([[1, 0, 1], [0, 1, 0]])
    y_pred = np.array([[1, 0, 0], [0, 1, 0]])
    multilabel_metrics(y_true, y_pred)
    # {'hamming_loss': 0.166..., 'jaccard_score': 0.833..., 'jaccard_score_zd1': 0.833...,
    #  'exact_match_ratio': 0.5, 'restricted_jaccard': 0.833...}
"""

from sklearn.metrics import hamming_loss, jaccard_score


def exact_match_ratio(y_true, y_pred):
    """
    Fraction of rows where the full predicted label vector exactly matches
    the true label vector (all labels correct simultaneously).

    Example:
        y_true = np.array([[1, 0, 1], [0, 1, 0]])
        y_pred = np.array([[1, 0, 0], [0, 1, 0]])
        exact_match_ratio(y_true, y_pred)
        # 0.5
    """
    return (y_true == y_pred).all(axis=1).mean()


def restricted_jaccard(y_true, y_pred):
    """
    Jaccard score (average='samples'), restricted to rows where y_true has
    at least one positive label -- excludes all-zero true rows, since
    Jaccard is degenerate/uninformative there.

    Example:
        y_true = np.array([[1, 0, 1], [0, 0, 0]])
        y_pred = np.array([[1, 0, 0], [0, 1, 0]])
        restricted_jaccard(y_true, y_pred)
        # 0.5
    """
    mask = y_true.any(axis=1)
    if not mask.any():
        print("restricted_jaccard: no rows with a positive true label, returning None")
        return None

    return jaccard_score(y_true[mask], y_pred[mask], average='samples')


def multilabel_metrics(y_true, y_pred):
    return {
        'hamming_loss': hamming_loss(y_true, y_pred),
        'jaccard_score': jaccard_score(y_true, y_pred, average='samples'),
        'jaccard_score_zd1': jaccard_score(y_true, y_pred, average='samples', zero_division=1),
        'exact_match_ratio': exact_match_ratio(y_true, y_pred),
        'restricted_jaccard': restricted_jaccard(y_true, y_pred),
    }
