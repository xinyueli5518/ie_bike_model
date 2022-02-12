import pandas as pd


def load_train_data(filename):
    """returns a pandas DataFrame with the contents of `hour.csv`"""
    """copy the data from `reference/` to `src/ie_bike_model/data/hour.csv`"""

    path = f"data/{filename}"
    df = pd.read_csv(path, parse_dates=["dteday"])
    X = df.drop(columns=["instant", "cnt", "casual", "registered"])
    y = df["cnt"]

    return X, y
