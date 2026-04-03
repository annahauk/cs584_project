import pandas as pd

def get_label_list(file_path):
    df = pd.read_csv(file_path)
    return df["logical_fallacies"].unique().tolist()

if __name__ == "__main__":
    label_list = get_label_list("data/all_logic_train.csv")
    print(label_list)