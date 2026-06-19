import os
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def main():
    train_csv_path = os.path.join("data", "processed", "train2_features.csv")
    scaler_out_path = os.path.join("weights", "scaler.pkl")

    if not os.path.exists(train_csv_path):
        print(f"Error: Could not find training data at {train_csv_path}")
        return

    print(f"Reading training dataset from {train_csv_path} (524MB)...")
    train_df = pd.read_csv(train_csv_path)

    print("Fitting MinMaxScaler...")
    stat_scaler = MinMaxScaler()
    # Features are from column 129 to 151 (lexical/anatomical/statistical index range)
    stat_scaler.fit(train_df.iloc[:, 129:151].values.astype(float))

    print(f"Saving fitted scaler to {scaler_out_path}...")
    os.makedirs("weights", exist_ok=True)
    with open(scaler_out_path, "wb") as f:
        pickle.dump(stat_scaler, f)
    print("Scaler successfully saved to weights/scaler.pkl")

if __name__ == "__main__":
    main()
