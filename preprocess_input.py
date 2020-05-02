import numpy as np
import math

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
            x_list.append(savgol_filter(X[x][i:i+step,0].ravel(), 25, 5).reshape(-1,1))
        x_list_c = np.concatenate([np.array(x_list).reshape(X.shape[1],1),np.array(X[x][:,1:])],axis=1)
        X_train_smooth.append(x_list_c)
        
    return np.array(X_train_smooth)

def buildData(
        df_tmp,
        interval=14, # length of input in day unit for encoder
        timestep = 85, # length of value in one daystep
        next_day=0 # output day that we want to predict
    ):
    df_X = [] # Encoder input list
    df_X2 = [] # Decoder input list
    df_y= [] # y_true value list
    start = 365 # start at second year
    end = (df_tmp.shape[0]-timestep-next_day)

    for day in range(start,end):

        d_features = circularIndex(range(day,day+(interval)+next_day+1))
        d_present = df_tmp[day:day+interval].reshape(-1,1) 
        d_past = df_tmp[day-(365):day-((365-interval))].reshape(-1,1) 

        df_X.append(np.concatenate([d_present,d_features[:-timestep*(next_day+1)]], axis=1))
        df_X2.append(np.concatenate([df_tmp[day+interval-365+next_day],d_features[-timestep:]],axis=1))

        df_y.append(df_tmp[day+(timestep+next_day),:,:])
    df_X = np.array(df_X)
    df_X2 = np.array(df_X2)
    df_y = np.array(df_y)
    return df_X, df_X2, df_y