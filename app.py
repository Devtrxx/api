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
    print(data)
    label2 = load('ride_prediction_encoder.joblib')
    scaler2 = load('ride_prediction_scaler.joblib')
    
    # model2 = tf.keras.models.load_model('ride_prediction.keras')
    layer1_w=np.array([[-0.21214743,  1.1613784 ,  0.7099438 , -0.3306076 ,  0.34989294,
            0.48632836, -0.5431796 ,  0.46742344, -0.74298483, -0.07896633],
        [ 0.16651022,  0.04402597, -0.32206765,  0.07432754,  0.64017886,
            -0.52495736, -0.53124034,  0.30130368,  0.43460345,  0.50068885],
        [ 1.2660726 , -0.18256542, -0.2969465 ,  0.95919406,  0.6283153 ,
            -0.00222168, -0.01387351,  0.5924543 ,  0.14866367, -1.3716631 ],
        [-0.06206094, -0.29248336, -0.9124618 ,  0.8181619 , -0.4028377 ,
            -0.78799075, -0.9698877 , -0.10285226, -0.8408352 , -0.40301436],
        [ 0.8872027 ,  0.70200413, -0.46186513,  0.80706257,  0.17877115,
            -0.30292124,  0.6687594 ,  0.7258183 , -0.2236675 , -0.9905398 ],
        [-0.21391535,  0.38555726, -0.01978815,  0.2036568 ,  0.14965592,
            -0.26133993,  0.3826898 ,  1.1459177 , -0.25228453, -0.0687854 ],
        [-0.15346241, -0.7838149 , -0.6105887 ,  0.23000523, -0.9342834 ,
            0.7365364 , -0.5488968 ,  0.09810582,  0.02703505, -0.23103124]],dtype='f')
    layer1_b=np.array([ 0.45004192, -0.8089768 , -0.3882554 ,  0.39315256,  0.23077857,
            0.68038094,  0.41155392,  0.08817519, -0.2849622 ,  0.5689967 ],dtype='f')
    layer2_w=np.array([[-0.37486303,  0.7721135 , -1.0156087 ,  0.8291489 , -0.12089685],
        [-0.3450488 , -1.0272747 , -0.08251603, -1.0431595 , -0.75805366],
        [-0.6513124 , -0.9867012 ,  0.06850582,  0.30724552, -0.95942044],
        [-0.22575425,  0.324649  ,  0.35334602, -0.77833605,  0.15663727],
        [ 0.2649791 ,  0.3182682 ,  0.5060977 , -1.1408677 ,  0.67693543],
        [-0.10274842,  0.35029507, -1.5661201 , -1.1877439 ,  0.05587848],
        [ 0.07665319,  0.8910436 ,  0.5331887 ,  0.21364552,  0.19083512],
        [ 0.10460491,  0.01172037, -0.4096023 ,  1.2482361 ,  0.4310244 ],
        [-0.40606293, -0.49817288,  0.7890781 ,  0.83848053, -0.6327891 ],
        [-0.5037162 ,  1.264642  ,  0.5135607 ,  0.88763434,  1.0482752 ]],dtype='f')
    layer2_b=np.array([-0.2420408 ,  0.4239888 ,  0.03111705, -0.02761324,  0.53409934],dtype='f')
    layer3_w=np.array([[-0.7276457],
        [ 1.4719024],
        [ 1.0089593],
        [-2.8366258],
        [ 1.0301241]],dtype='f')
    layer3_b=np.array([0.34049132],dtype='f')
    # print(model2.summary())

    src = np.array([data['from']])
    dest = np.array([data['to']])
    veh = np.array([data['carType']])
    src = label2.transform(src)
    dest = label2.transform(dest)
    veh = label.transform(veh)

    date,month,_ = data['date'].split('/')
    date = np.array([date])
    month = np.array([month])

    hour,minute = data['time'].split(':')

    hour = np.array([hour])
    minute = np.array([minute])
    print(src,dest,veh,date,hour,minute,month)

    X_test = np.c_[src,dest,veh,date,hour,minute,month]
    print(X_test.shape,X_test)
    X_norm = scaler2.transform(X_test)
    print(X_norm.shape,X_norm)
    predict = model(X_norm,layer1_w,layer2_w,layer3_w,layer1_b,layer2_b,layer3_b)[0,0]
    print(predict.shape,predict)
    return round(predict)

def layer_output(X,W,b):
    tmp=np.matmul(X,W)+b
    return np.maximum(0,tmp)
def model(X,W1,W2,W3,b1,b2,b3):
    A1=layer_output(X,W1,b1)
    A2=layer_output(A1,W2,b2)
    A3=layer_output(A2,W3,b3)
    return A3

    

