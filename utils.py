import pickle
import numpy as np

def get_road_adj(filename):
    with open(filename, 'rb') as fo:
        result = pickle.load(fo)
    road_info = result['road_links']
    road_dict_road2id = result['road_dict_road2id']
    num_roads = len(result['road_dict_id2road'])
    adjacency_matrix = np.zeros((int(num_roads), int(num_roads)), dtype=np.float32)
    adj_phase = np.full((int(num_roads), int(num_roads)), 'XX')
    for inter_dic in road_info:
        for link_dic in road_info[inter_dic]:
            source = link_dic[0]
            target = link_dic[1]
            type_p = link_dic[2]
            direction = link_dic[3]
            if type_p == 'go_straight':
                if direction == 0:
                    adj_phase[road_dict_road2id[source]][road_dict_road2id[target]] = 'WS'
                elif direction == 1:
                    adj_phase[road_dict_road2id[source]][road_dict_road2id[target]] = 'SS'
                elif direction == 2:
                    adj_phase[road_dict_road2id[source]][road_dict_road2id[target]] = 'ES'
                else:
                    adj_phase[road_dict_road2id[source]][road_dict_road2id[target]] = 'NS'
            elif type_p == 'turn_left':
                if direction == 0:
                    adj_phase[road_dict_road2id[source]][road_dict_road2id[target]] = 'WL'
                elif direction == 1:
                    adj_phase[road_dict_road2id[source]][road_dict_road2id[target]] = 'SL'
                elif direction == 2:
                    adj_phase[road_dict_road2id[source]][road_dict_road2id[target]] = 'EL'
                else:
                    adj_phase[road_dict_road2id[source]][road_dict_road2id[target]] = 'NL'
            else:
                if direction == 0:
                    adj_phase[road_dict_road2id[source]][road_dict_road2id[target]] = 'WR'
                elif direction == 1:
                    adj_phase[road_dict_road2id[source]][road_dict_road2id[target]] = 'SR'
                elif direction == 2:
                    adj_phase[road_dict_road2id[source]][road_dict_road2id[target]] = 'ER'
                else:
                    adj_phase[road_dict_road2id[source]][road_dict_road2id[target]] = 'NR'
            adjacency_matrix[road_dict_road2id[source]][road_dict_road2id[target]] = 1
    return adjacency_matrix, adj_phase

def robust_mape(pred, true, eps=1.0, min_true=0.5, use_median=True):
    """
    Calculate robust MAPE: only for true values > min_true, denominator plus eps, use median or mean.
    Args:
        pred (np.ndarray): Predicted values.
        true (np.ndarray): Ground truth values.
        eps (float): Small value to avoid division by zero.
        min_true (float): Minimum true value to consider.
        use_median (bool): If True, use median, else mean.
    Returns:
        float: Robust MAPE value (percentage).
    """
    mask = np.abs(true) > min_true
    ape = np.abs(pred - true) / (np.abs(true) + eps)
    ape = ape[mask]
    if len(ape) == 0:
        return np.nan
    if use_median:
        return np.median(ape) * 100
    else:
        return np.mean(ape) * 100

def robust_rmse(pred, true, min_true=0.5):
    """
    Calculate robust RMSE: only for true values > min_true.
    Args:
        pred (np.ndarray): Predicted values.
        true (np.ndarray): Ground truth values.
        min_true (float): Minimum true value to consider.
    Returns:
        float: Robust RMSE value.
    """
    mask = np.abs(true) > min_true
    err = (pred[mask] - true[mask]) ** 2
    if len(err) == 0:
        return np.nan
    return np.sqrt(np.mean(err))

def denorm_with_static(arr, mean, std, feat_idx):
    """
    Denormalize array for selected feature indices, keep static features unchanged.
    Args:
        arr (np.ndarray): Input array (..., F).
        mean (np.ndarray): Mean values for features.
        std (np.ndarray): Std values for features.
        feat_idx (list): Indices of dynamic features to denormalize.
    Returns:
        np.ndarray: Denormalized array.
    """
    arr_denorm = arr.copy()
    arr_denorm[..., :len(feat_idx)] = arr[..., :len(feat_idx)] * std[feat_idx] + mean[feat_idx]
    return arr_denorm

def robust_mae(pred, true, use_median=False, clip_percentile=None, min_true=1.0):
    mask = np.abs(true) > min_true
    err = np.abs(pred - true)
    err = err[mask]
    if clip_percentile is not None:
        threshold = np.percentile(err, clip_percentile)
        err = np.clip(err, None, threshold)
    if use_median:
        return np.median(err)
    else:
        return np.mean(err) 