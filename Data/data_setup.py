"""Pull LOGIC data from Hugging Face and save the CSV files"""
import pandas as pd
import os

def pull_logic_data():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    splits = {'train': 'data/train-00000-of-00001-8c3d4e48fe0f561b.parquet', 'test': 'data/test-00000-of-00001-ce92752fd4455cd1.parquet', 'dev': 'data/dev-00000-of-00001-99b3373cde156b17.parquet'}
    train_df = pd.read_parquet("hf://datasets/tasksource/logical-fallacy/" + splits["train"])
    test_df = pd.read_parquet("hf://datasets/tasksource/logical-fallacy/" + splits["test"])
    dev_df = pd.read_parquet("hf://datasets/tasksource/logical-fallacy/" + splits["dev"])
    train_df.to_csv(os.path.join(script_dir, "all_logic_train.csv"), index=False)
    test_df.to_csv(os.path.join(script_dir, "all_logic_test.csv"), index=False)
    dev_df.to_csv(os.path.join(script_dir, "all_logic_dev.csv"), index=False)

def separate_edu_climate_data():
    # Get the directory where this script is located to reliably create the edu_data and climate_data directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    edu_data_dir = os.path.join(script_dir, "edu_data")
    climate_data_dir = os.path.join(script_dir, "climate_data")
    if not os.path.exists(edu_data_dir):
        os.makedirs(edu_data_dir)
    if not os.path.exists(climate_data_dir):
        os.makedirs(climate_data_dir)
    # Separate the train data into edu and climate subsets
    all_train_df = pd.read_csv(os.path.join(script_dir, "all_logic_train.csv"))
    all_dev_df = pd.read_csv(os.path.join(script_dir, "all_logic_dev.csv"))
    all_test_df = pd.read_csv(os.path.join(script_dir, "all_logic_test.csv"))
    edu_train_df = all_train_df[all_train_df["config"] == "edu"]
    climate_train_df = all_train_df[all_train_df["config"] == "climate"]
    edu_train_df.to_csv(os.path.join(edu_data_dir, "logic_edu_train.csv"), index=False)
    climate_train_df.to_csv(os.path.join(climate_data_dir, "logic_climate_train.csv"), index=False)
    # Separate the dev data into edu and climate subsets
    edu_dev_df = all_dev_df[all_dev_df["config"] == "edu"]
    climate_dev_df = all_dev_df[all_dev_df["config"] == "climate"]
    edu_dev_df.to_csv(os.path.join(edu_data_dir, "logic_edu_dev.csv"), index=False)
    climate_dev_df.to_csv(os.path.join(climate_data_dir, "logic_climate_dev.csv"), index=False)
    # Separate the test data into edu and climate subsets
    edu_test_df = all_test_df[all_test_df["config"] == "edu"]
    climate_test_df = all_test_df[all_test_df["config"] == "climate"]
    edu_test_df.to_csv(os.path.join(edu_data_dir, "logic_edu_test.csv"), index=False)
    climate_test_df.to_csv(os.path.join(climate_data_dir, "logic_climate_test.csv"), index=False)


if __name__ == "__main__":    
    pull_logic_data()
    separate_edu_climate_data()