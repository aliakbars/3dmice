"""
Data loader for PhysioNet2012 from TSDB.

Handles conversion from long format to 3D tensor format suitable for imputation.
"""

import numpy as np
import pandas as pd
from tsdb import load
from pygrinder import mcar


# Static features to exclude from time series
STATIC_FEATURES = ["Age", "Gender", "Height", "ICUType", "Weight"]

# All time-varying features
TIME_VARYING_FEATURES = [
    "ALP", "ALT", "AST", "Albumin", "BUN", "Bilirubin", "Cholesterol",
    "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT",
    "HR", "K", "Lactate", "MAP", "MechVent", "Mg", "NIDiasABP", "NIMAP",
    "NISysABP", "Na", "PaCO2", "PaO2", "Platelets", "RespRate", "SaO2",
    "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC", "pH"
]


def load_physionet2012(
    subset="all",
    rate=0.1,
    seed=42,
    n_timesteps=48,
):
    """
    Load and preprocess PhysioNet2012 dataset from TSDB.
    
    Parameters
    ----------
    subset : str, default="all"
        Which subset to use: 'all', 'set-a', 'set-b', or 'set-c'.
    rate : float, default=0.1
        Rate of artificially introduced missing values for evaluation.
    seed : int, default=42
        Random seed for reproducibility.
    n_timesteps : int, default=48
        Number of time steps (hours) per patient. PhysioNet2012 uses first 48h.
        
    Returns
    -------
    dict with keys:
        - 'train_X': Training data with missing values
        - 'train_X_ori': Original training data (ground truth)
        - 'train_indicating_mask': Mask for artificially masked values
        - 'test_X': Test data with missing values  
        - 'test_X_ori': Original test data
        - 'test_indicating_mask': Mask for test masked values
    """
    np.random.seed(seed)
    
    # Load raw data from TSDB
    raw_data = load("physionet_2012")
    
    # Determine which subsets to use
    if subset == "all":
        subsets = ["set-a", "set-b", "set-c"]
    else:
        subsets = [subset]
    
    # Combine all requested subsets
    dfs = [raw_data[s] for s in subsets if s in raw_data]
    df = pd.concat(dfs, ignore_index=True)
    
    # Convert to 3D tensor format
    X_3d = _convert_to_3d(df, n_timesteps)
    
    # Split into train/test (80/20)
    n_samples = X_3d.shape[0]
    n_train = int(0.8 * n_samples)
    
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    train_X_ori = X_3d[train_idx]
    test_X_ori = X_3d[test_idx]
    
    # Apply MCAR missing pattern for evaluation
    train_X, train_mask = _apply_missing(train_X_ori, rate, seed)
    test_X, test_mask = _apply_missing(test_X_ori, rate, seed + 1)
    
    return {
        "train_X": train_X,
        "train_X_ori": train_X_ori,
        "train_indicating_mask": train_mask,
        "test_X": test_X,
        "test_X_ori": test_X_ori,
        "test_indicating_mask": test_mask,
    }


def _convert_to_3d(df, n_timesteps):
    """
    Convert long-format DataFrame to 3D tensor.
    
    Parameters
    ----------
    df : pd.DataFrame
        Long-format data with RecordID, Time, and feature columns.
    n_timesteps : int
        Number of time steps per sample.
        
    Returns
    -------
    X : ndarray of shape (n_samples, n_timesteps, n_features)
    """
    # Get unique record IDs
    record_ids = df["RecordID"].unique()
    n_samples = len(record_ids)
    n_features = len(TIME_VARYING_FEATURES)
    
    # Initialize output tensor with NaN
    X = np.full((n_samples, n_timesteps, n_features), np.nan)
    
    # Process each record
    for i, rec_id in enumerate(record_ids):
        rec_df = df[df["RecordID"] == rec_id]
        
        for _, row in rec_df.iterrows():
            # Parse time - format is "HH:MM" 
            time_val = row["Time"]
            if pd.isna(time_val):
                continue
            
            # Convert time to hour index (0-47)
            if isinstance(time_val, str):
                try:
                    hour = int(time_val.split(":")[0])
                except (ValueError, AttributeError):
                    continue
            else:
                hour = int(time_val)
            
            if hour >= n_timesteps:
                hour = n_timesteps - 1  # Cap at max timestep
            
            # Fill in feature values
            for j, feat in enumerate(TIME_VARYING_FEATURES):
                if feat in row.index and pd.notna(row[feat]):
                    X[i, hour, j] = row[feat]
    
    return X


def _apply_missing(X, rate, seed):
    """
    Apply additional MCAR missing pattern to data.
    
    Only applies to values that are already observed (not NaN).
    
    Parameters
    ----------
    X : ndarray
        Original data.
    rate : float
        Missing rate to apply.
    seed : int
        Random seed.
        
    Returns
    -------
    X_masked : ndarray
        Data with additional missing values.
    indicating_mask : ndarray
        Boolean mask where True indicates artificially masked values.
    """
    np.random.seed(seed)
    
    X_masked = X.copy()
    
    # Get indices of observed values
    observed_mask = ~np.isnan(X)
    n_observed = observed_mask.sum()
    
    if n_observed == 0:
        return X_masked, np.zeros_like(X, dtype=bool)
    
    # Create mask for values to artificially mask
    indicating_mask = np.zeros_like(X, dtype=bool)
    
    # Randomly select observed values to mask
    observed_indices = np.where(observed_mask)
    n_to_mask = int(rate * n_observed)
    
    if n_to_mask > 0:
        mask_indices = np.random.choice(len(observed_indices[0]), n_to_mask, replace=False)
        
        for idx in mask_indices:
            i, j, k = observed_indices[0][idx], observed_indices[1][idx], observed_indices[2][idx]
            indicating_mask[i, j, k] = True
            X_masked[i, j, k] = np.nan
    
    return X_masked, indicating_mask


def get_evaluation_data(data, split="test"):
    """
    Extract data needed for imputation evaluation.
    
    Parameters
    ----------
    data : dict
        Data dictionary from load_physionet2012.
    split : str, default="test"
        Which split to use: 'train' or 'test'.
        
    Returns
    -------
    X_incomplete : ndarray
        Data with missing values.
    X_complete : ndarray
        Ground truth data for masked values.
    indicating_mask : ndarray
        Boolean mask where True indicates artificially masked values.
    """
    prefix = split
    
    X_incomplete = data[f"{prefix}_X"]
    X_complete = data[f"{prefix}_X_ori"]
    indicating_mask = data[f"{prefix}_indicating_mask"]
    
    return X_incomplete, X_complete, indicating_mask
