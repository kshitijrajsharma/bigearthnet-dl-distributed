"""Convert BigEarthNet TIF files from S3 to TFRecord format for efficient training."""

import argparse
import os
import tempfile
import boto3
import numpy as np
import tensorflow as tf
from rasterio.io import MemoryFile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm

def read_s3_tif(s3_path):
    """Download and read GeoTIFF from S3."""
    s3_client = boto3.client('s3')
    bucket, key = s3_path.replace('s3://', '').split('/', 1)
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    with MemoryFile(obj['Body'].read()) as memfile:
        with memfile.open() as dataset:
            return dataset.read()

def pad_to_size(array, target_shape, pad_value=0):
    """Pad 2D array to target shape with zeros."""
    pad_width = [(0, max(0, target - current)) 
                 for target, current in zip(target_shape, array.shape)]
    return np.pad(array, pad_width, mode='constant', constant_values=pad_value)

def process_patch(row_dict, target_size=(120, 120)):
    """Download, process, and combine S1, S2, and label data for a single patch."""
    try:
        # Build S3 paths for all bands
        s2_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        s3_paths = {
            's1_vv': f"{row_dict['s1_path']}/{row_dict['s1_name']}_VV.tif",
            's1_vh': f"{row_dict['s1_path']}/{row_dict['s1_name']}_VH.tif",
            'label': f"{row_dict['reference_path']}/{row_dict['patch_id']}_reference_map.tif"
        }
        for band in s2_bands:
            s3_paths[f's2_{band}'] = f"{row_dict['s2_path']}/{row_dict['patch_id']}_{band}.tif"
        
        # Download all files in parallel (15 concurrent requests)
        file_data = {}
        with ThreadPoolExecutor(max_workers=15) as executor:
            future_to_key = {executor.submit(read_s3_tif, path): key for key, path in s3_paths.items()}
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                file_data[key] = future.result()[0]  # Get first band
        
        # Stack S1 bands (VV, VH) -> (120, 120, 2)
        s1_data = np.stack([
            pad_to_size(file_data['s1_vv'], target_size),
            pad_to_size(file_data['s1_vh'], target_size)
        ], axis=-1).astype(np.float32)
        
        # Stack S2 bands (12 bands) -> (120, 120, 12)
        s2_data = np.stack([
            pad_to_size(file_data[f's2_{band}'], target_size) for band in s2_bands
        ], axis=-1).astype(np.float32)
        
        # Process label map -> (120, 120)
        label = pad_to_size(file_data['label'], target_size).astype(np.uint8)
        
        return {
            'patch_id': row_dict['patch_id'],
            's1_data': s1_data,
            's2_data': s2_data,
            'label': label,
        }
    except Exception as e:
        print(f"Error processing {row_dict['patch_id']}: {e}")
        return None

def sample_stratified(df, fraction):
    """Sample dataset maintaining train/val/test split proportions."""
    if fraction >= 1.0:
        return df
    if 'split' not in df.columns:
        return df.sample(frac=fraction, random_state=42)
    return df.groupby('split', group_keys=False).apply(
        lambda x: x.sample(frac=fraction, random_state=42), include_groups=False
    ).reset_index(drop=True)

def serialize_example(patch_id, s1_data, s2_data, label):
    """Serialize arrays to TFRecord format."""
    feature = {
        'patch_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[patch_id.encode()])),
        's1_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[s1_data.tobytes()])),
        's2_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[s2_data.tobytes()])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tobytes()])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

def convert_files(metadata_path, output_path, fraction=1.0, workers=10, batch_size=100):
    """Convert BigEarthNet metadata to TFRecord files."""
    # Load metadata
    import pyarrow.parquet as pq
    df = pq.read_table(metadata_path).to_pandas()
    print(f"Total patches: {len(df)}")
    
    # Sample if needed
    if fraction < 1.0:
        df = sample_stratified(df, fraction)
        print(f"Sampled {len(df)} patches ({fraction*100:.1f}% stratified by split)")
    
    print(f"Processing {len(df)} patches with {workers} workers")
    
    # Create output directory (local or S3)
    is_s3 = output_path.startswith('s3://')
    if not is_s3:
        os.makedirs(output_path, exist_ok=True)
    
    # Process in batches
    records = df.to_dict('records')
    file_num = 0
    
    for i in tqdm(range(0, len(records), batch_size), desc="Processing batches"):
        batch = records[i:i+batch_size]
        
        # Download and process patches in parallel
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = [f.result() for f in 
                      [executor.submit(process_patch, row) for row in batch] 
                      if f.result()]
        
        if not results:
            continue
        
        # Write batch to TFRecord
        output_file = f"{output_path}/part-{file_num:05d}.tfrecord"
        
        if is_s3:
            # Write to temp file then upload to S3
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tfrecord') as tmp:
                with tf.io.TFRecordWriter(tmp.name) as writer:
                    for r in results:
                        writer.write(serialize_example(r['patch_id'], r['s1_data'], r['s2_data'], r['label']))
                
                # Upload to S3
                s3_client = boto3.client('s3')
                bucket, key = output_file.replace('s3://', '').split('/', 1)
                s3_client.upload_file(tmp.name, bucket, key)
                os.unlink(tmp.name)
        else:
            with tf.io.TFRecordWriter(output_file) as writer:
                for r in results:
                    writer.write(serialize_example(r['patch_id'], r['s1_data'], r['s2_data'], r['label']))
        
        file_num += 1
    
    print(f"\nDataset saved to {output_path}")
    print(f"Total files: {file_num}")

def main():
    parser = argparse.ArgumentParser(description='Convert BigEarthNet TIF files to TFRecord format')
    parser.add_argument('--meta', required=True, help='S3 path to metadata parquet file')
    parser.add_argument('--out', required=True, help='Output directory for TFRecord files')
    parser.add_argument('--frac', type=float, default=1.0, help='Data fraction (0.0-1.0), stratified by split')
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel workers')
    parser.add_argument('--batch', type=int, default=100, help='Patches per output file')
    
    args = parser.parse_args()
    
    if not 0 < args.frac <= 1.0:
        raise ValueError("--frac must be between 0.0 and 1.0")
    
    convert_files(args.meta, args.out, args.frac, args.workers, args.batch)

if __name__ == "__main__":
    main()
