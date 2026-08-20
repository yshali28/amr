"""
Multi-label evaluation metrics.

Example:
    import numpy as np
    from src.evaluation.metrics import multilabel_metrics

    y_true = np.array([[1, 0, 1], [0, 1, 0]])
    y_pred = np.array([[1, 0, 0], [0, 1, 0]])
    multilabel_metrics(y_true, y_pred)
    # {'hamming_loss': 0.166..., 'jaccard_score': 0.833...}
"""

from sklearn.metrics import hamming_loss, jaccard_score


def multilabel_metrics(y_true, y_pred):
    return {
        'hamming_loss': hamming_loss(y_true, y_pred),
        'jaccard_score': jaccard_score(y_true, y_pred, average='samples'),
    }
