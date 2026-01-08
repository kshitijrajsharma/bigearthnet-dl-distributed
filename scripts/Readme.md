# BigEarthNet Data Pipeline

Four-stage pipeline: metadata generation, validation, conversion, and training.

## Scripts

1. **gen_metadata** - Add S3 paths to metadata
2. **check** - Validate file existence
3. **to_peta_parquet** - Convert TIF to Parquet format
4. **train** - Train model with TensorFlow

## Usage

### Script 1: Generate Metadata

```bash
uv run scripts/gen_metadata.py --meta s3://ubs-datasets/bigearthnet/metadata.parquet --out s3://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet
```

### Script 2: Check S3 Files

```bash
# Check all files
uv run scripts/check.py --meta s3://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet --out s3://ubs-homes/erasmus/raj/dlproject/results.json --workers 50

# Check 1% sample
uv run scripts/check.py --meta s3://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet --out results_1pct.json --frac 0.01
```

### Script 3: Convert to Parquet

```bash
# Convert all data
uv run scripts/to_peta_parquet.py --meta s3://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet --out s3://ubs-homes/erasmus/raj/dlproject/petastorm_data --workers 10

# Convert 1% sample
uv run scripts/to_peta_parquet.py --meta s3://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet --out s3://ubs-homes/erasmus/raj/dlproject/petastorm_data_1pct --frac 0.01
```

### Script 4: Train Model

```bash
# Train with local Parquet data
uv run scripts/train.py --data /path/to/parquet_data --epochs 10 --batch 32 --lr 0.001

# Train with S3 Parquet data
uv run scripts/train.py --data s3://ubs-homes/erasmus/raj/dlproject/petastorm_data_1pct --epochs 5 --batch 16 --save model.keras
```

## Arguments

**Common:**
- `--meta`: S3 path to metadata parquet
- `--out`: S3 or local path for output
- `--frac`: Fraction of data (0.0-1.0), stratified by split (default: 1.0)

**check:**
- `--workers`: Parallel workers (default: 50)

**to_peta_parquet:**
- `--workers`: Parallel workers (default: 10)
- `--batch`: Batch size (default: 100)

**train:**
- `--data`: Path to Parquet data (local or S3)
- `--epochs`: Number of epochs (default: 10)
- `--batch`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--save`: Path to save trained model

## Workflow

```bash
# 1. Generate metadata
uv run scripts/gen_metadata.py --meta INPUT --out OUTPUT

# 2. Check files
uv run scripts/check.py --meta METADATA --out RESULTS

# 3. Convert to Parquet
uv run scripts/to_peta_parquet.py --meta METADATA --out DATA --frac 0.01

# 4. Train model
uv run scripts/train.py --data DATA --epochs 5 --batch 16
```

Use `--frac 0.01` for testing, `--frac 1.0` for full dataset.

