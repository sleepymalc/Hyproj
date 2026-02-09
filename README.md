# A Unified Theory of Random Projection for Influence Functions

Experimental framework to validate theoretical claims about regularized sketching for Influence Functions.

## Setup

It's **not** required to follow the exact same steps in this section. But this is a verified environment setup flow that may help users to avoid most of the issues during the installation.

```bash
conda create -n hyproj python=3.10
conda activate hyproj

conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip install sjlt --no-build-isolation
pip install dattri

pip install -r requirements.txt
```

## Experiments

### 1. Spectrum Bounds Validation (`spectrum_bounds.py`)

**Purpose:** Validate that sketched influence scores approximate exact scores within theoretical error bounds.

**What it measures:**
- Two test modes:
  - **Self-influence** (`--test_mode self`): Computes `ratio = sketched_score / exact_score` for training gradients
  - **Cross-influence** (`--test_mode test`): Computes normalized additive error `ε = |B̃_λ - B_λ| / (√φ₀(g) × √φ₀(v))` for test gradients
- `ε_95 = 95th percentile of error` across multiple random projection trials

**Usage:**
```bash
# Basic run (self-influence mode)
python experiments/spectrum_bounds.py --dataset mnist --model mlp

# Cross-influence mode with test set gradients
python experiments/spectrum_bounds.py --dataset mnist --model mlp --test_mode test

# With more trials for statistical significance
python experiments/spectrum_bounds.py --dataset mnist --model mlp --num_trials 10

# Different projection type
python experiments/spectrum_bounds.py --proj_type sjlt

# Large model (disk-cached gradients)
python experiments/spectrum_bounds.py --dataset maestro --model musictransformer --offload disk

# Custom m and λ sweep ranges (powers of 2 for m, powers of 10 for λ)
python experiments/spectrum_bounds.py --m_exp_min 5 --m_exp_max 18 --lambda_exp_min -6 --lambda_exp_max 0
```

**Key Arguments:**
| Argument           | Description                                      | Default |
| ------------------ | ------------------------------------------------ | ------- |
| `--test_mode`      | Test mode: `self` or `test`                      | self    |
| `--num_test_grads` | Number of test gradients to use                  | 500     |
| `--num_trials`     | Number of random projection trials               | 5       |
| `--min_d_lambda`   | Skip λ values where d_λ < this threshold         | 1.0     |
| `--m_exp_min`      | Minimum m exponent (m = 2^exp)                   | 2       |
| `--m_exp_max`      | Maximum m exponent (m = 2^exp)                   | 20      |
| `--m_steps`        | Number of log-spaced m values (0 = powers of 2)  | 0       |
| `--lambda_exp_min` | Minimum λ exponent (λ = 10^exp)                  | -8      |
| `--lambda_exp_max` | Maximum λ exponent (λ = 10^exp)                  | 2       |
| `--lambda_steps`   | Number of log-spaced λ values (0 = powers of 10) | 0       |

### 2. Hyperparameter Selection (`hyperparam_selection.py`)

**Purpose:** Compare theory-driven vs utility-driven hyperparameter selection strategies.

**What it measures:**
- LDS (Linear Datamodeling Score) as a function of λ and m
- Optimal λ*(m) that maximizes validation LDS for each m
- Test LDS for the selected hyperparameters
- Whether m ≥ d_λ* is sufficient for good downstream utility

**Two approaches:**
1. **Theory-driven:** Choose m based on effective dimension d_λ (e.g., m ≈ d_λ / ε²)
2. **Utility-driven:** Sweep λ to maximize validation LDS, then verify m ≥ d_λ*

**Usage:**
```bash
# Basic run with validation/test split
python experiments/hyperparam_selection.py --dataset mnist --model mlp

# With more test samples and trials
python experiments/hyperparam_selection.py --dataset mnist --model mlp \
    --num_test_grads 1000 --num_trials 10

# Custom validation ratio and m/λ sweeps
python experiments/hyperparam_selection.py --dataset mnist --model mlp \
    --val_ratio 0.15 --m_exp_min 6 --m_exp_max 18

# Large model with disk offload
python experiments/hyperparam_selection.py --dataset maestro --model musictransformer \
    --offload disk --batch_size 8
```

**Key Arguments:**
| Argument           | Description                                      | Default |
| ------------------ | ------------------------------------------------ | ------- |
| `--val_ratio`      | Fraction of test set for validation              | 0.1     |
| `--num_test_grads` | Number of test samples (before val/test split)   | 500     |
| `--num_trials`     | Number of random projection trials               | 5       |
| `--m_exp_min`      | Minimum m exponent (m = 2^exp)                   | 5       |
| `--m_exp_max`      | Maximum m exponent (m = 2^exp)                   | 20      |
| `--m_steps`        | Number of log-spaced m values (0 = powers of 2)  | 0       |
| `--lambda_exp_min` | Minimum λ exponent (λ = 10^exp)                  | -8      |
| `--lambda_exp_max` | Maximum λ exponent (λ = 10^exp)                  | 2       |
| `--lambda_steps`   | Number of log-spaced λ values (0 = powers of 10) | 0       |

### 3. Faithfulness-Utility Alignment (`faithfulness_utility.py`)

**Purpose:** Investigate whether maximizing downstream utility (LDS) leads to selecting hyperparameters in the empirically "unfaithful" region where sketched scores deviate significantly from exact scores.

**Key Questions:**
1. **Alignment:** For fixed m, does λ*(m) = argmax LDS(m, λ) fall in the unfaithful region?
2. **Monotonicity:** Does the optimal utility LDS(m, λ*(m)) increase with m?

**What it measures:**
- Faithfulness: Normalized bilinear form error ε = |B̃_λ - B_λ| / (√φ₀(g) × √φ₀(v))
- Utility: LDS on validation set (for λ selection) and test set (for final evaluation)
- Per-m analysis: λ*(m), faithfulness at λ*(m), and test LDS

**Usage:**
```bash
# Basic run
python experiments/faithfulness_utility.py --dataset mnist --model mlp

# With stricter faithfulness threshold
python experiments/faithfulness_utility.py --dataset mnist --model mlp \
    --faithfulness_threshold 0.05

# With more test samples and trials
python experiments/faithfulness_utility.py --dataset mnist --model mlp \
    --num_test_samples 1000 --num_trials 10

# Custom validation split and m/λ sweeps
python experiments/faithfulness_utility.py --dataset cifar2 --model resnet9 \
    --val_ratio 0.15 --m_exp_min 8 --m_exp_max 18 --lambda_exp_min -6 --lambda_exp_max 0
```

**Key Arguments:**
| Argument                   | Description                                         | Default |
| -------------------------- | --------------------------------------------------- | ------- |
| `--faithfulness_threshold` | Threshold for empirical faithfulness (ε_95)         | 0.01    |
| `--num_test_samples`       | Number of test samples for faithfulness measurement | 500     |
| `--num_trials`             | Number of random projection trials                  | 5       |
| `--val_ratio`              | Fraction of test set for validation                 | 0.1     |
| `--m_exp_min`              | Minimum m exponent (m = 2^exp)                      | 5       |
| `--m_exp_max`              | Maximum m exponent (m = 2^exp)                      | 20      |
| `--m_steps`                | Number of log-spaced m values (0 = powers of 2)     | 0       |
| `--lambda_exp_min`         | Minimum λ exponent (λ = 10^exp)                     | -8      |
| `--lambda_exp_max`         | Maximum λ exponent (λ = 10^exp)                     | 2       |
| `--lambda_steps`           | Number of log-spaced λ values (0 = powers of 10)    | 0       |

## Common Command-Line Arguments

These arguments are shared across all experiments:

| Argument       | Description                                  | Default                          |
| -------------- | -------------------------------------------- | -------------------------------- |
| `--dataset`    | Dataset: mnist, cifar2, maestro              | mnist                            |
| `--model`      | Model: lr, mlp, resnet9, musictransformer    | mlp                              |
| `--proj_type`  | Projection: normal, rademacher, sjlt         | normal                           |
| `--batch_size` | GPU batch size (tune for memory/utilization) | 32 (16 for hyperparam_selection) |
| `--offload`    | Gradient storage: none, cpu, disk            | cpu                              |
| `--cache_dir`  | Directory for disk cache                     | ./grad_cache                     |
| `--output_dir` | Directory for results                        | ./results                        |
| `--seed`       | Random seed                                  | 42                               |
| `--device`     | cuda or cpu                                  | cuda                             |

## Model/Dataset Configurations

| Model            | Dataset | Parameters | Recommended Settings                |
| ---------------- | ------- | ---------- | ----------------------------------- |
| lr               | mnist   | ~8K        | `--offload none` or `--offload cpu` |
| mlp              | mnist   | ~100K      | `--offload cpu`                     |
| resnet9          | cifar2  | ~6M        | `--offload cpu`                     |

