# Results

This directory is for storing analysis results, plots, and reports derived from experiments.

##  Organization

```
results/
├── plots/       
│   ├── speedup_preprocess.png   
│   ├── speedup_training.png 
```

## Working with Profile Data

Profile JSON files from experiments can be analyzed to understand:
- Conversion throughput and bottlenecks
- Training speed and GPU utilization
- Data loading vs computation time
- Memory usage patterns

Example analysis:
```python
import json
import pandas as pd

# Load conversion profile
with open('path/to/conversion_profile.json') as f:
    profile = json.load(f)
    
# Extract metrics
metrics = {
    'total_duration': profile['summary']['total_duration'],
    'train_samples': profile['summary']['train_samples'],
    'dataset_size_gb': profile['summary']['dataset_size_gb']
}
```

## Notebooks

Use Jupyter notebooks in the `notebooks/` directory for interactive analysis and visualization of results.

## Comparing Experiments

To compare different configurations:
1. Extract metrics from profile JSON files
2. Create comparison tables or plots
3. Document findings in this directory

This helps identify optimal configurations for:
- Number of Spark executors
- Data fraction for rapid prototyping
- GPU count for training
- Batch size and learning rate