# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides training and test datasets.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--raw_data", type=str, required=True, help="Path to raw data")
    parser.add_argument("--train_data", type=str, required=True, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test dataset")
    parser.add_argument(
        "--test_train_ratio",
        type=float,
        default=0.2,
        help="Test-train ratio",
    )
    return parser.parse_args()


def main(args):
    """Read, preprocess, split, and save datasets."""
    if not 0 < args.test_train_ratio < 1:
        raise ValueError("--test_train_ratio must be greater than 0 and less than 1")

    data = pd.read_csv(args.raw_data)
    required_columns = {
        "Segment",
        "Kilometers_Driven",
        "Mileage",
        "Engine",
        "Power",
        "Seats",
        "price",
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Raw data is missing required columns: {sorted(missing_columns)}")
    if data[list(required_columns)].isna().any().any():
        raise ValueError("Raw data contains missing values in required columns")

    data = data.copy()
    data["Segment"] = LabelEncoder().fit_transform(data["Segment"])
    train_data, test_data = train_test_split(
        data, test_size=args.test_train_ratio, random_state=42
    )

    train_path = Path(args.train_data)
    test_path = Path(args.test_data)
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    train_data.to_csv(train_path / "train.csv", index=False)
    test_data.to_csv(test_path / "test.csv", index=False)

    mlflow.log_metrics(
        {"train_row_count": len(train_data), "test_row_count": len(test_data)}
    )


if __name__ == "__main__":
    args = parse_args()
    print(f"Raw data path: {args.raw_data}")
    print(f"Train dataset output path: {args.train_data}")
    print(f"Test dataset path: {args.test_data}")
    print(f"Test-train ratio: {args.test_train_ratio}")
    with mlflow.start_run():
        main(args)
