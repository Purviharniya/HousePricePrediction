import pickle
import json
import numpy as np
__locations = None
__data_cols = None
__model = None

def get_loc_names():
    return __locations

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_cols.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_cols))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)

def load_saved():
    global __data_cols
    global __locations
    global __model

    with open("./artifacts/columns.json",'r') as f:
        __data_cols = json.load(f)['data_columns']
        __locations = __data_cols[3:]

    with open("./artifacts/house_price_predict_model.pickle",'rb') as f:
        __model = pickle.load(f)

if __name__=="__main__":
    load_saved()
    print(get_loc_names())
    print(get_estimated_price('Other',1000,2,2))
 