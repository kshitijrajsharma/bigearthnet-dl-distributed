# Experiments

This directory stores experiment outputs and results from conversion and training runs.

## Structure

When you run the pipeline using bash scripts, results are organized as follows:

```
experiments/
└── {experiment_name}/
    └── petastorm/
        ├── frac_0.01/          # 1% of data
        │   ├── exec_8/         # Converted with 8 executors
        │   │   ├── train/
        │   │   ├── validation/
        │   │   ├── test/
        │   │   └── profile/
        │   │       ├── conversion_profile.json
        │   │       ├── conversion_profile.log
        │   │       ├── train_gpu4_profile.json
        │   │       ├── train_gpu4_profile.log
        │   │       └── ... (profiles for other GPU counts)
        │   ├── exec_5/
        │   ├── exec_2/
        │   └── exec_1/
        ├── frac_0.05/          # 5% of data
        ├── frac_0.10/          # 10% of data
        └── frac_0.20/          # 20% of data
```

## Purpose

This structure enables:
- **Performance analysis**: Compare conversion times across different executor counts
- **Scalability testing**: Evaluate training performance with varying data sizes
- **GPU scaling studies**: Test multi-GPU training efficiency (1, 2, 3, 4 GPUs)
- **Reproducibility**: Keep complete records of each experimental run

## Profile Files

Each experiment includes:
- **conversion_profile.json/log**: Spark configuration, processing times, data statistics
- **train_profile.json/log**: GPU usage, training metrics, evaluation results

## Running Experiments

```bash
# Create a new experiment
./scripts/bash/convert.sh my_experiment client
./scripts/bash/train.sh my_experiment train 8

# Results will be stored in:
# experiments/my_experiment/petastorm/...
```

## Note on S3 Storage

By default, experiments are stored on S3. The bash scripts are configured to write to:
```
s3://ubs-homes/erasmus/raj/dlproject/experiments/
```

Update the `ROOT_DIR` variable in `scripts/bash/convert.sh` and `scripts/bash/train.sh` to use your own S3 bucket or local storage.