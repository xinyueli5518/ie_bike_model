import numpy as np
import pandas as pd
import os

from ie_bike_model.persistence import persist_model
from ie_bike_model.persistence import load_model
from ie_bike_model.data import load_train_data

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import (
    ColumnTransformer,
    make_column_transformer,
    make_column_selector,
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.pipeline import FeatureUnion, make_union


def train_and_persist():
    """trains the RandomForestRegressor model displayed in the notebook,
    using hour.csv as training data,
    and saves it to a `model.joblib` using joblib"""

    # load data
    X, y = load_train_data("hour.csv")

    # data preprocessing
    ffiller = FunctionTransformer(ffill_missing)

    weather_enc = make_pipeline(
        ffiller,
        OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=X["weathersit"].nunique()
        ),
    )

    ct = make_column_transformer(
        (ffiller, make_column_selector(dtype_include=np.number)),
        (weather_enc, ["weathersit"]),
    )

    preprocessing = FeatureUnion(
        [
            ("is_weekend", FunctionTransformer(is_weekend)),
            ("year", FunctionTransformer(year)),
            ("column_transform", ct),
        ]
    )

    # train model
    reg = Pipeline(
        [("preprocessing", preprocessing), ("model", RandomForestRegressor())]
    )

    # train test split
    X_train, y_train = X.loc[X["dteday"] < "2012-10"], y.loc[X["dteday"] < "2012-10"]
    X_test, y_test = X.loc["2012-10" <= X["dteday"]], y.loc["2012-10" <= X["dteday"]]

    # fit model
    reg.fit(X_train, y_train)

    # persist model
    persist_model(reg, "model.joblib")


def predict(dteday, hr, weathersit, temp, atemp, hum, windspeed, model):
    """receives the input parameters (see example in README.md),
    preprocesses the input data (dummy variables, scaling, ...),
    passes that to the trained model, and returns the number of expected users"""

    # preprocess test data
    X_test = pd.DataFrame(
        [
            [
                pd.to_datetime(dteday),
                hr,
                weathersit,
                temp,
                atemp,
                hum,
                windspeed,
            ]
        ],
        columns=["dteday", "hr", "weathersit", "temp", "atemp", "hum", "windspeed"],
    )

    # check existing model and load
    if len(os.listdir("models")) == 0:  ## if no existing model
        train_and_persist()
    else:
        loaded_model = load_model(model)

    # predict number of users
    y_pred = loaded_model.predict(X_test)

    return y_pred


def ffill_missing(ser):
    return ser.fillna(method="ffill")


def is_weekend(data):
    return data["dteday"].dt.day_name().isin(["Saturday", "Sunday"]).to_frame()


def year(data):
    # Our reference year is 2011, the beginning of the training dataset
    return (data["dteday"].dt.year - 2011).to_frame()
