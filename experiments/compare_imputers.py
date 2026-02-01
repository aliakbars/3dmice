"""
Experiment script to compare imputation methods on PhysioNet2012.

Compares:
- 3D-MICE (cross-sectional MICE + longitudinal GP)
- LOCF (Last Observation Carried Forward)
- Median imputation
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from tabulate import tabulate

from data_loader import load_physionet2012, get_evaluation_data
from evaluation import evaluate_imputation
from imputers.locf import LOCFImputer
from imputers.median import MedianImputer
from imputers.mice_3d import MICE3D


def run_experiment(
    subset="all",
    rate=0.1,
    pattern="point",
    seed=42,
    max_samples=None,
):
    """
    Run imputation comparison experiment.
    
    Parameters
    ----------
    subset : str
        PhysioNet2012 subset to use.
    rate : float
        Missing data rate.
    pattern : str
        Missing data pattern.
    seed : int
        Random seed.
    max_samples : int, optional
        Limit number of samples for faster testing.
    """
    print("=" * 60)
    print("3D-MICE Imputation Comparison on PhysioNet2012")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Subset: {subset}")
    print(f"  Missing rate: {rate}")
    print(f"  Missing pattern: {pattern}")
    print(f"  Random seed: {seed}")
    
    # Load data
    print("\n[1/5] Loading PhysioNet2012 dataset...")
    data = load_physionet2012(subset=subset, rate=rate, seed=seed)
    
    # Get test data for evaluation
    X_incomplete, X_ground_truth, indicating_mask = get_evaluation_data(data, split="test")
    
    # Limit samples if specified
    if max_samples is not None and max_samples < len(X_incomplete):
        print(f"  Limiting to {max_samples} samples for faster testing")
        X_incomplete = X_incomplete[:max_samples]
        X_ground_truth = X_ground_truth[:max_samples]
        indicating_mask = indicating_mask[:max_samples]
    
    print(f"  Data shape: {X_incomplete.shape}")
    print(f"  Total missing values in original: {np.isnan(X_incomplete).sum()}")
    print(f"  Artificially masked values: {indicating_mask.sum()}")
    
    # Initialize imputers
    imputers = {
        "LOCF": LOCFImputer(),
        "Median": MedianImputer(),
        "3D-MICE": MICE3D(
            mice_max_iter=10,
            random_state=seed,
        ),
    }
    
    results = {}
    
    # Run each imputer
    for name, imputer in imputers.items():
        print(f"\n[{list(imputers.keys()).index(name) + 2}/5] Running {name}...")
        start_time = time.time()
        
        try:
            X_imputed = imputer.fit_transform(X_incomplete)
            elapsed = time.time() - start_time
            
            # Evaluate
            metrics = evaluate_imputation(X_imputed, X_ground_truth, indicating_mask)
            metrics["Time (s)"] = elapsed
            results[name] = metrics
            
            print(f"  Completed in {elapsed:.2f}s")
            print(f"  MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, MRE: {metrics['MRE']:.4f}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {"MAE": np.nan, "RMSE": np.nan, "MRE": np.nan, "Time (s)": np.nan}
    
    # Print results table
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    
    # Format as table
    headers = ["Method", "MAE", "RMSE", "MRE", "Time (s)"]
    rows = []
    for name, metrics in results.items():
        rows.append([
            name,
            f"{metrics['MAE']:.4f}",
            f"{metrics['RMSE']:.4f}",
            f"{metrics['MRE']:.4f}",
            f"{metrics['Time (s)']:.2f}",
        ])
    
    try:
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    except ImportError:
        # Fallback if tabulate not installed
        print(f"{'Method':<12} {'MAE':<10} {'RMSE':<10} {'MRE':<10} {'Time (s)':<10}")
        print("-" * 52)
        for row in rows:
            print(f"{row[0]:<12} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<10}")
    
    # Determine best method
    best_mae = min(results.items(), key=lambda x: x[1].get("MAE", np.inf))
    best_rmse = min(results.items(), key=lambda x: x[1].get("RMSE", np.inf))
    
    print(f"\nBest MAE: {best_mae[0]} ({best_mae[1]['MAE']:.4f})")
    print(f"Best RMSE: {best_rmse[0]} ({best_rmse[1]['RMSE']:.4f})")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare imputation methods on PhysioNet2012")
    parser.add_argument("--subset", type=str, default="all", help="Dataset subset")
    parser.add_argument("--rate", type=float, default=0.1, help="Missing rate")
    parser.add_argument("--pattern", type=str, default="point", help="Missing pattern")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples (for testing)")
    
    args = parser.parse_args()
    
    run_experiment(
        subset=args.subset,
        rate=args.rate,
        pattern=args.pattern,
        seed=args.seed,
        max_samples=args.max_samples,
    )
