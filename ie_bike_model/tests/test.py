import os
from ie_bike_model.ai import train_and_persist
from ie_bike_model.ai import predict



# test return type of predict()
def predict_return_positive_int():
    y_pred = predict(
    dteday="2012-11-01",
    hr=10,
    weathersit="Clear, Few clouds, Partly cloudy, Partly cloudy",
    temp=0.3,
    atemp=0.31,
    hum=0.8,
    windspeed=0.0,
)
    return isinstance(y_pred, int) and y_pred > 0 

# test file created by train_and_persist()
def test_train_and_persist_create_model():
    before = len(os.listdir("models"))
    train_and_persist()
    after = len(os.listdir("models"))
    
    return before == after -1
    