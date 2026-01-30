"""Add S3 paths (s1_path, s2_path, reference_path) to BigEarthNet metadata."""

import argparse

import pyarrow.compute as pc
import pyarrow.parquet as pq


def generate_metadata_with_paths(input_path, output_path):
    """Add S3 file paths to BigEarthNet metadata for efficient data access.

    Why needed: Original metadata only contains patch IDs. This pre-computes full S3 paths
    to avoid string concatenation during training, improving data loading performance.

    The S3 structure follows BigEarthNet's folder hierarchy:
    - S1 patches are grouped by first 5 underscore-separated fields
    - S2/reference patches are grouped by first 6 underscore-separated fields
    """
    print(f"Reading metadata from {input_path}")
    table = pq.read_table(input_path)

    # Extract folder prefix from s1_name (first 5 fields)
    # Example: S1A_IW_GRDH_1SDV_20170613T165043_33UUP_33_62 -> S1A_IW_GRDH_1SDV_20170613T165043
    s1_prefix = pc.replace_substring_regex(
        table["s1_name"], r"((?:[^_]+_){4}[^_]+)_.*", r"\1"
    )
    s1_path = pc.binary_join_element_wise(
        "s3://ubs-datasets/bigearthnet/BigEarthNet-S1/", s1_prefix, ""
    )
    s1_path = pc.binary_join_element_wise(s1_path, table["s1_name"], "/")

    # Extract folder prefix from patch_id (first 6 fields)
    # Example: S2A_MSIL2A_20170613T101031_33_62_patch123 -> S2A_MSIL2A_20170613T101031_33_62
    s2_prefix = pc.replace_substring_regex(
        table["patch_id"], r"((?:[^_]+_){5}[^_]+)_.*", r"\1"
    )
    s2_path = pc.binary_join_element_wise(
        "s3://ubs-datasets/bigearthnet/BigEarthNet-S2/", s2_prefix, ""
    )
    s2_path = pc.binary_join_element_wise(s2_path, table["patch_id"], "/")

    ref_path = pc.binary_join_element_wise(
        "s3://ubs-datasets/bigearthnet/Reference_Maps/", s2_prefix, ""
    )
    ref_path = pc.binary_join_element_wise(ref_path, table["patch_id"], "/")

    table = table.append_column("s1_path", s1_path)
    table = table.append_column("s2_path", s2_path)
    table = table.append_column("reference_path", ref_path)

    print(f"Writing enhanced metadata to {output_path}")
    pq.write_table(table, output_path)
    print(f"Done! {len(table)} rows with s1_path, s2_path, reference_path")


def main():
    parser = argparse.ArgumentParser(
        description="Generate BigEarthNet metadata with S3 paths"
    )
    parser.add_argument(
        "--meta",
        default="s3://ubs-datasets/bigearthnet/metadata.parquet",
        help="S3 path to input metadata parquet",
    )
    parser.add_argument(
        "--out",
        default="s3://ubs-homes/erasmus/raj/dlproject/metadata_with_paths.parquet",
        help="S3 path for output metadata with paths",
    )
    args = parser.parse_args()
    generate_metadata_with_paths(args.meta, args.out)


if __name__ == "__main__":
    main()
