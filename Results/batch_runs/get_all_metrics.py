import pandas as pd
from pathlib import Path

def get_all_metrics(base_path="."):
    """
    Collect and aggregate only aggregate_metrics_mean_std.csv files
    from each run folder.
    """
    metrics_frames = []
    base_path = Path(base_path)
    target_file = "aggregate_metrics_mean_std.csv"
    
    # Scan all subdirectories and read only the aggregate summary file.
    for run_dir in sorted(base_path.glob("*/")): 
        if not run_dir.is_dir():
            continue
            
        run_name = run_dir.name
        metrics_path = run_dir / target_file
        if not metrics_path.exists():
            continue

        try:
            df = pd.read_csv(metrics_path)
            df['run'] = run_name
            df['metrics_path'] = str(metrics_path)
            metrics_frames.append(df)
        except Exception as e:
            print(f"Error reading {metrics_path}: {e}")
    
    # Aggregate and save
    if metrics_frames:
        combined_df = pd.concat(metrics_frames, ignore_index=True)
        combined_df.to_csv('aggregated_metrics_combined.csv', index=False)
        print(f"Aggregated {len(metrics_frames)} files named {target_file}")
        return combined_df

    print(f"No {target_file} files found under {base_path}")
    return pd.DataFrame()

if __name__ == "__main__":
    metrics = get_all_metrics(".")