import numpy as np

def r2(true, pred, mask=None, clip=False):
    """
    R^2 (coefficient of determination).
    """
    true, pred = np.asarray(true), np.asarray(pred)
    if mask is not None:
        mask = np.asarray(mask)
        true = true[mask]
        pred = pred[mask]
    mean_true = np.mean(true)
    ss_tot = np.sum((true - mean_true) ** 2)
    ss_res = np.sum((true - pred) ** 2)
    r2 = 1 - ss_res / ss_tot
    if clip: r2 = np.clip(r2, 0.0, 1.0)
    return r2


def p_spearman(true, pred, mask=None):
    """
    Spearman's rank correlation coefficient
    """
    true, pred = np.asarray(true), np.asarray(pred)
    if mask is not None:
        mask = np.asarray(mask)
        true = true[mask]
        pred = pred[mask]
    rank_true = np.argsort(np.argsort(true))
    rank_pred = np.argsort(np.argsort(pred))
    diff = rank_true - rank_pred
    rho = 1 - 6 * np.sum(diff ** 2) / (len(diff) ** 3 - len(diff))
    return rho
    

def p_pearson(true, pred, mask=None):
    """
    Pearson correlation coefficient
    """
    true, pred = np.asarray(true), np.asarray(pred)
    if mask is not None:
        mask = np.asarray(mask)
        true = true[mask]
        pred = pred[mask]
    mean_true = np.mean(true)
    mean_pred = np.mean(pred)
    cov = np.mean((true - mean_true) * (pred - mean_pred))
    std_true = np.std(true)
    std_pred = np.std(pred)
    rho = cov / (std_true * std_pred)
    return rho


def mae(true, pred, mask=None):
    """
    Mean Absolute Error
    """
    true, pred = np.asarray(true), np.asarray(pred)
    if mask is not None:
        mask = np.asarray(mask)
        true = true[mask]
        pred = pred[mask]
    mae = np.mean(np.abs(true - pred))
    return mae

def rmse(true, pred, mask=None):
    """
    Root Mean Squared Error
    """
    true, pred = np.asarray(true), np.asarray(pred)
    if mask is not None:
        mask = np.asarray(mask)
        true = true[mask]
        pred = pred[mask]
    rmse = np.sqrt(np.mean((true - pred) ** 2))
    return rmse


def smape(true, pred, mask=None):
    """
    Symmetric Mean Absolute Percentage Error
    """
    true, pred = np.asarray(true), np.asarray(pred)
    if mask is not None:
        mask = np.asarray(mask)
        true = true[mask]
        pred = pred[mask]
    smape = 2 * np.mean(np.abs(true - pred) / (np.abs(true) + np.abs(pred) + 1e-9))
    return smape


def mape(true, pred, mask=None):
    """
    Mean Absolute Percentage Error
    """
    true, pred = np.asarray(true), np.asarray(pred)
    if mask is not None:
        mask = np.asarray(mask)
        true = true[mask]
        pred = pred[mask]
    mape = np.mean(np.abs(true - pred) / (np.abs(true) + 1e-9))
    return mape