#!/bin/bash

# Fractions to run
fractions=(0.01 0.03 0.05 0.07)

# Output base directory
output_base="s3://ubs-homes/erasmus/raj/dlproject/experiments/petastorm"

# Spark parameters
executor_mem="16g"
driver_mem="8g"
core="3"
n_executor="8"
spark_packages="ch.cern.sparkmeasure:spark-measure_2.12:0.27"

for frac in "${fractions[@]}"; do
    out_dir="${output_base}/frac_${frac}"
    echo "Running conversion with fraction: ${frac}, output dir: ${out_dir}"
    spark-submit \
        --master yarn \
        --deploy-mode client \
        --packages "${spark_packages}" \
        scripts/to_petastorm.py \
        --out "${out_dir}" \
        --frac "${frac}" \
        --executor-mem "${executor_mem}" \
        --driver-mem "${driver_mem}" \
        --core "${core}" \
        --n_executor "${n_executor}" \
        > /dev/null 2>&1
    echo "Completed conversion for fraction: ${frac}"
done