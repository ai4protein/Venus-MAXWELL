import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import numpy as np

@torch.no_grad()
def evaluate(L, Y, M):
    x = L[M].flatten().cpu().numpy()
    y = Y[M].flatten().cpu().numpy()
    return {
        "PC": pearsonr(x,y)[0],
        "SC": spearmanr(x,y)[0],
    }

# @torch.no_grad()
# def reg_evaluate(L, Y, M):
#     x = L[M].flatten().cpu().numpy()
#     y = Y[M].flatten().cpu().numpy()
#     return {
#         "R-MSE": mean_squared_error(x, y),
#         "R-MAE": mean_absolute_error(x, y),
#         "R-MAPE": mean_absolute_percentage_error(x, y),
#         "R-PC": pearsonr(x, y)[0],
#         "R-SC": spearmanr(x, y)[0],
#     }

# @torch.no_grad()
# def cls_evaluate(L, Y, M):
#     x = L[M].flatten().cpu().numpy()
#     y = Y[M].flatten().cpu().numpy()
#     y = (y > 0).astype(int)
#     # 检测是否全为1个类：
#     if np.all(y == 1) or np.all(y == 0):
#         return {
#             "B-ROAUC": None,
#             "B-AUPRC": None,
#             "B-F1": None,
#         }
#     return {
#         "B-ROAUC": roc_auc_score(y, x),
#         "B-AUPRC": average_precision_score(y, x),
#         "B-F1": f1_score(y, (x> 0).astype(int)),
#     }