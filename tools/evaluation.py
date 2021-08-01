import numpy as np

def evaluation_metrics(y_pred, y):
    y_pred_cat = 1 * (y_pred > 0.5)
    correct = np.sum(y_pred_cat == y)
    predictions = len(y)
    true_pos = np.sum((y_pred_cat == 1) & (y == 1))
    positives = np.sum(y)
    pred_pos = np.sum(y_pred_cat)
    accuracy = correct / predictions
    recall = true_pos / positives
    precision = true_pos / pred_pos
    f1 = 2 * (recall * precision) / (recall + precision)
    return accuracy, recall, precision, f1

def multi_evaluation_metrics(y_pred, y):
    y_pred = np.argmax(y_pred, axis=1)
    correct = np.sum(y_pred == y)
    predictions = len(y)
    accuracy = correct / predictions
    return accuracy