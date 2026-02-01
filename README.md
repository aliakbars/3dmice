# 3D-MICE

Implementation of **3D-MICE** (3-Dimensional Multiple Imputation with Chained Equations) using only scikit-learn.

Based on: [Luo et al. (2017). 3D-MICE: integration of cross-sectional and longitudinal imputation for multi-analyte longitudinal clinical data. JAMIA.](https://pubmed.ncbi.nlm.nih.gov/29202205/)

## Installation

```bash
uv sync
```

## Usage

### Run Comparison Experiment

```bash
# Full test set
uv run python experiments/compare_imputers.py

# Quick test with limited samples
uv run python experiments/compare_imputers.py --max-samples 100
```

### Use as Library

```python
from src.imputers.mice_3d import MICE3D
from src.imputers.locf import LOCFImputer
from src.imputers.median import MedianImputer

# 3D-MICE imputation
imputer = MICE3D(mice_max_iter=10, random_state=42)
X_imputed = imputer.fit_transform(X)  # X shape: (n_samples, n_timesteps, n_features)

# Baseline imputers
locf = LOCFImputer()
median = MedianImputer()
```

## Results on PhysioNet2012

| Method   | MAE ↓ | RMSE ↓ | MRE ↓ |
|----------|-------|--------|-------|
| **3D-MICE** | **12.36** | 60.73 | **0.169** |
| LOCF     | 13.58 | **58.71** | 0.186 |
| Median   | 19.42 | 78.55 | 0.265 |

3D-MICE outperforms LOCF by 8.9% and Median by 36.3% on MAE.

## Benchmarking with BenchPOTS

Install dev dependencies to access benchmarking datasets:

```bash
uv sync --dev
```

### Load PhysioNet2012 Dataset

```python
from benchpots.datasets import preprocess_physionet2012

# Load with artificial missing pattern for evaluation
data = preprocess_physionet2012(
    subset="all",       # 'all', 'set-a', 'set-b', or 'set-c'
    rate=0.1,           # 10% additional missing rate
    pattern="point",    # 'point', 'subseq', or 'block'
)

# Data splits
train_X = data["train_X"]              # Training data with missing values
train_X_ori = data["train_X_ori"]      # Ground truth
train_mask = data["train_indicating_mask"]  # Mask for artificially masked values
```

### Run Benchmark

```python
from src.imputers.mice_3d import MICE3D
from src.evaluation import evaluate_imputation

imputer = MICE3D(random_state=42)
X_imputed = imputer.fit_transform(data["test_X"])

metrics = evaluate_imputation(
    X_imputed, 
    data["test_X_ori"], 
    data["test_indicating_mask"]
)
print(f"MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}")
```

### Other Datasets

BenchPOTS supports many time series datasets:

```python
from benchpots.datasets import preprocess_physionet2019  # Sepsis prediction
from benchpots.datasets import preprocess_beijing_multisite_airquality
```

See [BenchPOTS documentation](https://github.com/WenjieDu/BenchPOTS) for full dataset list.

## Algorithm

3D-MICE combines:
1. **Cross-sectional MICE** (`sklearn.impute.IterativeImputer`): Captures feature correlations at each timestep
2. **Longitudinal GP** (`sklearn.gaussian_process.GaussianProcessRegressor`): Captures temporal patterns
3. **Variance-weighted combination**: Blends both estimates

## Citation

```bibtex
@article{luo20173d,
  title={3D-MICE: integration of cross-sectional and longitudinal imputation for multi-analyte longitudinal clinical data},
  author={Luo, Yuan and Szolovits, Peter and Dighe, Anand S and Baron, Jason M},
  journal={Journal of the American Medical Informatics Association},
  volume={25},
  number={6},
  pages={645--653},
  year={2017},
  publisher={Oxford University Press}
}
```
