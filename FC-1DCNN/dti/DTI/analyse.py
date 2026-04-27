import numpy as np
import logging

LOG = logging.getLogger('analyse')


def analyse_classification(loss, target_list, pred_list):
    target_list = np.asarray(target_list).astype(int)
    pred_list = np.asarray(pred_list).astype(int)

    TP = ((pred_list == 1) & (target_list == 1)).sum()
    TN = ((pred_list == 0) & (target_list == 0)).sum()
    FP = ((pred_list == 1) & (target_list == 0)).sum()
    FN = ((pred_list == 0) & (target_list == 1)).sum()

    acc = round((TP + TN) / pred_list.size, 4)

    precision = round(TP / (TP + FP), 4) if (TP + FP) > 0 else 0.0
    recall = round(TP / (TP + FN), 4) if (TP + FN) > 0 else 0.0
    specificity = round(TN / (TN + FP), 4) if (TN + FP) > 0 else 0.0
    loss = round(loss, 4)

    LOG.info(
        'Loss:{}, Correct:{}/{}(acc:{}), precision:{}, recall:{}, specificity:{}'
        .format(loss, TP + TN, pred_list.size, acc, precision, recall, specificity)
    )

    return {
        'loss': loss,
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'specificity': specificity
    }