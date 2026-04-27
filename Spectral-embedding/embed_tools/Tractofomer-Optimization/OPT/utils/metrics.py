import numpy as np


def get_mse(y_pred, y_target):
    non_zero_pos = y_target != 0
    return np.fabs((y_target[non_zero_pos] - y_pred[non_zero_pos])).mean()