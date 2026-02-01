"""
Evaluation metrics for imputation quality.
"""

import numpy as np


def mae(y_true, y_pred, mask=None):
    """
    Mean Absolute Error.
    
    Parameters
    ----------
    y_true : ndarray
        Ground truth values.
    y_pred : ndarray
        Predicted/imputed values.
    mask : ndarray, optional
        Boolean mask. If provided, only values where mask is True are evaluated.
        
    Returns
    -------
    float
        Mean absolute error.
    """
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred, mask=None):
    """
    Root Mean Square Error.
    
    Parameters
    ----------
    y_true : ndarray
        Ground truth values.
    y_pred : ndarray
        Predicted/imputed values.
    mask : ndarray, optional
        Boolean mask. If provided, only values where mask is True are evaluated.
        
    Returns
    -------
    float
        Root mean square error.
    """
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mre(y_true, y_pred, mask=None):
    """
    Mean Relative Error (normalized by sum of absolute ground truth values).
    
    Parameters
    ----------
    y_true : ndarray
        Ground truth values.
    y_pred : ndarray
        Predicted/imputed values.
    mask : ndarray, optional
        Boolean mask. If provided, only values where mask is True are evaluated.
        
    Returns
    -------
    float
        Mean relative error.
    """
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


def evaluate_imputation(X_imputed, X_ground_truth, indicating_mask):
    """
    Evaluate imputation quality on masked values.
    
    Parameters
    ----------
    X_imputed : ndarray
        Imputed data.
    X_ground_truth : ndarray
        Ground truth data.
    indicating_mask : ndarray
        Boolean mask where True indicates values to evaluate.
        
    Returns
    -------
    dict
        Dictionary with MAE, RMSE, and MRE metrics.
    """
    # Convert mask to boolean if needed
    mask = indicating_mask.astype(bool)
    
    return {
        "MAE": mae(X_ground_truth, X_imputed, mask),
        "RMSE": rmse(X_ground_truth, X_imputed, mask),
        "MRE": mre(X_ground_truth, X_imputed, mask),
    }
