# from fastapi import FastAPI, HTTPException
# from joblib import load
# import numpy as np

# app = FastAPI()

# model = load('sgd_regressor_model.joblib')
# label = load('label_encoder.joblib')
# scaler = load('scaler.joblib')

# @app.post("/calculate_distance")
# def calculate_distance(data: dict):
    # print("hello world!")
    # car_type = data['carType']
    # car = np.array([car_type])
    # print(car,car.shape)
    # main = label.transform(car)

    # try:
    #     km = "20 km"
    #     km = ((int)(km[:-2]))*1000
    #     kilo = np.array(km)

    #     X_train = np.c_[kilo,main].reshape(-1,1)
    #     X_norm = scaler.transform(X_train)
    #     return model.predict(X_norm)



    # except Exception as e:
    #     print(f"Error generating distance: {str(e)}")
from fastapi import FastAPI, HTTPException
from joblib import load
import numpy as np
import httpx
import requests

app = FastAPI()

model = load('sgd_regressor_model.joblib')
label = load('label_encoder.joblib')
scaler = load('scaler.joblib')

distance_from_maps = "http://192.168.50.67:8000/distance"

@app.post("/calculate_distance")
async def calculate_distance(data: dict):
    print(data)
    car_type = data['carType']
    car = np.array([car_type])
    main = label.transform(car)


    response = httpx.post(distance_from_maps, json=data)
    km = response.json()
    print(km)
    try:
        km = ((int)(km))
        kilo = np.array([km])
        print(main,main.shape,kilo.shape)

        X_train = np.c_[kilo, main].reshape(-1,2)
        print(X_train.shape)
        X_norm = scaler.transform(X_train)
        print(X_norm.shape)


        prediction = model.predict(X_norm)

        return {'og':str(round(prediction[0]))+"Rs",'distance':str(round(km/1000))+"km"}

    except Exception as e:
        print(f"Error generating distance: {str(e)}")
