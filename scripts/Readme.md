# BigEarthNet Pipeline

Complete workflow for BigEarthNet satellite imagery: metadata generation, validation, TFRecord conversion, and U-Net training.

## Workflow

### 1. Generate Metadata with S3 Paths

```bash
uv run scripts/gen_metadata.py \
  --meta s3://ubs-datasets/bigearthnet/metadata.parquet \
  --out s3://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet
```

### 2. Validate S3 Files

```bash
uv run scripts/check.py \
  --meta s3://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet \
  --out s3://ubs-homes/erasmus/raj/dlproject/validation.json \
  --frac 0.001 \
  --workers 50
```

### 3. Convert to TFRecord

```bash
# Test: 0.1% data (~480 patches)
uv run scripts/to_tfrecord.py \
  --meta s3://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet \
  --out s3://ubs-homes/erasmus/raj/dlproject/tfrecords_test \
  --frac 0.001

# Production: full dataset
uv run scripts/to_tfrecord.py \
  --meta s3://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet \
  --out s3://ubs-homes/erasmus/raj/dlproject/tfrecords_full \
  --workers 20
```

### 4. Train Model

```bash
# Quick test: 2 epochs
uv run python scripts/train.py --data ./data --epochs 2 --batch 8

# Production: 50 epochs with model save
uv run python scripts/train.py --data ./data --epochs 50 --batch 32 --save model.keras
```

## Parameters

**gen_metadata.py**
- `--meta`: S3 path to base metadata parquet
- `--out`: Output path for augmented metadata

**check.py**
- `--meta`: S3 path to metadata with file paths
- `--out`: Output path for validation results
- `--frac`: Data fraction for validation (default: 1.0)
- `--workers`: Parallel workers (default: 50)

**to_tfrecord.py**
- `--meta`: S3 path to metadata parquet
- `--out`: Output directory
- `--frac`: Data fraction (0.001-1.0, default: 1.0)
- `--workers`: Parallel workers (default: 10)
- `--batch`: Patches per file (default: 100)

**train.py**
- `--data`: TFRecord directory path
- `--epochs`: Training epochs (default: 10)
- `--batch`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--save`: Model output path (.keras)

## Data Scale

| Fraction | Patches |
|----------|---------|
| 0.001 | 480 |
| 0.01 | 4,800 |
| 0.1 | 48,000 |
| 1.0 | 480,038 |

## Format

**TFRecord**: `patch_id` (string), `s1_data` (120x120x2 float32), `s2_data` (120x120x12 float32), `label` (120x120 uint8)

**Model**: U-Net encoder-decoder, 120x120x14 input, 256 output classes, ~947K params
