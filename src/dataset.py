# src/dataset.py
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import RAW_DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE
from src.features import preprocess_data


def load_data():
    df = pd.read_csv(RAW_DATA_PATH)
    df = preprocess_data(df)
    return df


def split_data(df):
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
