import numpy as np
import math
from scipy.signal import savgol_filter

def circularIndex(
        day, # input data in specific length
        timestep = 85 # length of value in one day
    ):
    res = []
    for d in day:
        val = 2*math.pi*d/365
        res.extend([math.sin(val), math.cos(val)]*timestep)
    return np.array(res).reshape(-1,2)

def smooth(
        X, # input data as numpy array
        timestep = 85 # length of value in one daystep
    ):
    X_train_smooth = []
    for x in range(X.shape[0]):
        x_list = []
        for i in range(0,X.shape[1],timestep):
            x_list.append(savgol_filter(X[x][i:i+timestep,0].ravel(), 25, 5).reshape(-1,1))
        x_list_c = np.concatenate([np.array(x_list).reshape(X.shape[1],1),np.array(X[x][:,1:])],axis=1)
        X_train_smooth.append(x_list_c)
        
    return np.array(X_train_smooth)

def normalized(data):
    max_train = 1980.64
    return (data/ max_train)+1e-5

def buildData(
        df_tmp,
        df_tmp_2,
        interval = 14, # length of input in day unit for encoder
        timestep = 85, # length of value in one daystep
        next_day = 0 # output day that we want to predict
    ):
    df_tmp = normalized(np.array(df_tmp))
    df_tmp_2 = normalized(np.array(df_tmp_2))

    df_X = []
    df_X2 = []
    df_y= []
    start = 0
    end = (df_tmp.shape[0]-interval-next_day)

    for day in range(start,end):
        d_features = circularIndex(range(day,day+(interval)+next_day+1))
        d_present = df_tmp[day:day+interval].reshape(-1,1) 
        df_X.append(np.concatenate([d_present,d_features[:-timestep*(next_day+1)]], axis=1))
        df_X2.append(np.concatenate([df_tmp_2[day+interval+next_day],d_features[-timestep:]],axis=1))

        df_y.append(df_tmp[day+(interval+next_day),:,:])

    df_X = np.array(df_X)
    df_X2 = np.array(df_X2)
    df_y = np.array(df_y)
    return df_X, df_X2, df_y