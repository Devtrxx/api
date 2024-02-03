# from fastapi import FastAPI, HTTPException
# from joblib import load
# import numpy as np

# app = FastAPI()

# model = load('sgd_regressor_model.joblib')
# label = load('label_encoder.joblib')
# scaler = load('scaler.joblib')

# @app.post("/calculate_distance")
# def calculate_distance(data: dict):
#     car_type = data['cartype']
#     car = np.array(car_type)
#     main = label.transform(car)

#     try:
#         km = response["choices"][0]["message"]["content"]
#         km = ((int)(km[:-2]))*1000
#         kilo = np.array(km)

#         X_train = np.c_[kilo,main].reshape(-1,1)
#         X_norm = scaler.transform(X_train)
#         return model.predict(X_norm)



#     except Exception as e:
#         print(f"Error generating distance: {str(e)}")
from fastapi import FastAPI, HTTPException
from joblib import load
import numpy as np
import httpx 
import requests

app = FastAPI()

model = load('sgd_regressor_model.joblib')
label = load('label_encoder.joblib')
scaler = load('scaler.joblib')

distance_from_maps = "http://127.0.0.1:8001/distance_from_maps"

@app.post("/calculate_distance")
def calculate_distance(data: dict):
    car_type = data['carType']


    main = label.transform(np.array(car_type).reshape(-1, 1))


    data = {
        "from": data["from"], 
        "to": data["to"],  
    }

    response = requests.post(distance_from_maps, json=data)
    km = response.json()

    try:
        km = ((int)(km)) * 1000
        kilo = np.array(km)

        X_train = np.c_[kilo, main].reshape(-1, 2)
        X_norm = scaler.transform(X_train)


        prediction = model.predict(X_norm)

        return {"distance_prediction": prediction.item(), "km_from_openai": km}

    except Exception as e:
        print(f"Error generating distance: {str(e)}")
