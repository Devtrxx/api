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
import aiohttp
import asyncio
import tensorflow as tf

app = FastAPI()

model = load('sgd_regressor_model.joblib')
label = load('label_encoder.joblib')
scaler = load('scaler.joblib')

distance_from_maps = "http://192.168.9.67:8000/distance"

# async def make_request(data: dict):
#     distance_from_maps = "http://192.168.9.67:8000/distance"

#     async with aiohttp.ClientSession() as session:
#         async with session.post(distance_from_maps, json=data) as response:
#             return await response.json()

@app.post("/calculate_distance")
async def calculate_distance(data: dict):
    print(data)
    car_type = data['carType']
    car = np.array([car_type])
    main = label.transform(car)


    print(car)
    # timeout_seconds = 10
    response = httpx.post(distance_from_maps, json=data)
    # print(response)
    # result = await (make_request(data))
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
        print(prediction.shape,type(prediction),prediction)
        print(type(prediction[0]),prediction[0].shape)

        fare_l = (prediction[0]-prediction[0]*30/100)
        fare_u = (prediction[0]+prediction[0]*20/100)
        print(fare_l,fare_u)

        fare = (data['fare'])
        
        return {'fare_l':str(fare_l),'fare_u':str((fare_u)),'og':str((fare)),'distance':str((km/1000))}
    except Exception as e:
        print(f"Error generating distance: {str(e)}")

@app.post("/rides")
def rides(data: dict):
    label2 = load('ride prediction encoder.joblib')
    scaler2 = load('ride prediction scaler.joblib')
    
    model2 = tf.keras.models.load_model('ride prediction.keras')

    src = np.array([data['from']])
    dest = np.array([data['to']])
    veh = np.array([data['carType']])
    src = label2.transform(src)
    dest = label2.transform(dest)
    veh = label.transform(veh)

    date = np.array([data['date']])
    hour = np.array([data["hour"]])
    minute = np.array([data["minute"]])
    month = np.array([data["month"]])

    X_test = np.c_[src,dest,veh,date,hour,minute,month]
    X_norm = scaler2.tranform(X_test)
    predict = model2.predict(X_norm)
    return predict
