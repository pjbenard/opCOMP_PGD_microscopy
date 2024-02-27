import numpy as np
from scipy.spatial import distance_matrix


def pair_GT_estimation(t_true, t_esti, r=0.02, min_sep_gtr_r=True):
    dist_true_esti = distance_matrix(t_true, t_esti)

    if min_sep_gtr_r:
        args_pairs = np.argwhere(dist_true_esti < r)
        args_t_true_paired, args_t_esti_paired = args_pairs[:, 0], args_pairs[:, 1]

        t_true_paired = t_true[args_t_true_paired]
        t_esti_paired = t_esti[args_t_esti_paired]

        mask_true_not_paired = np.ones(t_true.shape[0], dtype=bool)
        mask_true_not_paired[args_t_true_paired] = False
        t_true_not_paired = t_true[mask_true_not_paired]

        mask_esti_not_paired = np.ones(t_esti.shape[0], dtype=bool)
        mask_esti_not_paired[args_t_esti_paired] = False
        t_esti_not_paired = t_esti[mask_esti_not_paired]

    return t_true_paired, t_true_not_paired, t_esti_paired, t_esti_not_paired


def get_sub_metrics(t_true_paired, t_true_not_paired, t_esti_paired, t_esti_not_paired):
    TP = t_esti_paired.shape[0]
    FP = t_esti_not_paired.shape[0]
    FN = t_true_not_paired.shape[0]
    return TP, FP, FN


def Jaccard(TP, FP, FN):
    # TP / (TP + FP + FN)
    return TP / (TP + FP + FN)


def Recall(TP, FP, FN):
    # TP / (TP + FN)
    return TP / (TP + FN)


def Precision(TP, FP, FN):
    # TP / (TP + FP)
    return TP / (TP + FP)


def compute_metrics(t_true_paired, t_true_not_paired, t_esti_paired, t_esti_not_paired):
    TP, FP, FN = get_sub_metrics(t_true_paired, t_true_not_paired, t_esti_paired, t_esti_not_paired)

    metrics = [Jaccard, Recall, Precision]
    metrics_res = {metric.__name__: metric(TP, FP, FN) for metric in metrics}

    return metrics_res


def RMSE_1d(t_true_paired, t_esti_paired):
    TP = t_true_paired.shape[0]
    return np.sqrt(np.sum((t_true_paired - t_esti_paired) ** 2) / TP)


def RMSE(t_true_paired, t_esti_paired, convert_to_nm=True):
    # RMSE nD
    nD = t_true_paired.shape[1]
    rmse = np.array([RMSE_1d(t_true_paired[:, d], t_esti_paired[:, d]) for d in range(nD)])
    if convert_to_nm:
        rmse *= 1000
    return rmse
