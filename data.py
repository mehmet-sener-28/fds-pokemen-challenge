import pandas as pd
from pathlib import Path
from .config import COL_TARGET, COL_ID, DATA_PATH

def load_data(data_path: Path | str = DATA_PATH):
    data_path = Path(data_path)
    train_df = pd.read_json(data_path / "train.jsonl", lines=True)
    test_df = pd.read_json(data_path / "test.jsonl", lines=True)

    print(f" Train: {train_df.shape[0]} battles")
    print(f" Test: {test_df.shape[0]} battles")

    assert COL_TARGET in train_df.columns, "Target missing!"
    assert COL_TARGET not in test_df.columns, "Target leakage!"

    return train_df, test_df


def clean_train(train_df: pd.DataFrame) -> pd.DataFrame:
    print("\nCleaning data...")

    flawed_indices = []

    if len(train_df) > 4877:
        flawed_indices.append(4877)

    if COL_ID in train_df.columns:
        flawed_by_id = train_df[train_df[COL_ID] == 4877].index.tolist()
        flawed_indices.extend(flawed_by_id)

    flawed_indices = list(set(flawed_indices))
    if flawed_indices:
        train_df = train_df.drop(index=flawed_indices).reset_index(drop=True)
        print(f" Removed {len(flawed_indices)} flawed row(s): {flawed_indices}")
        print(f" Train shape after cleaning: {train_df.shape}")
    else:
        print(" No flawed rows found")

    return train_df


def load_and_clean_data(data_path: Path | str = DATA_PATH):
    train_df, test_df = load_data(data_path)
    train_df = clean_train(train_df)
    return train_df, test_df