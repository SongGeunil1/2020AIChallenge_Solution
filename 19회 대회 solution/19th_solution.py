import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import os
from sklearn.model_selection import TimeSeriesSplit


# 데이터 read
train_proton = pd.read_csv("train/train_proton.csv")
test_proton = pd.read_csv("test/test_proton.csv")

train_AC_H1_EPM = pd.read_csv("train/train_AC_H1_EPM.csv")
test_AC_H1_EPM = pd.read_csv("test/test_AC_H1_EPM.csv")

train_xray = pd.read_csv("train/train_xray.csv")
test_xray = pd.read_csv("test/test_xray.csv")

train_AC_H0_SWE = pd.read_csv("train/train_AC_H0_SWE.csv")
test_AC_H0_SWE = pd.read_csv("test/test_AC_H0_SWE.csv")



for df in [train_AC_H1_EPM,train_AC_H0_SWE,train_proton,train_xray,test_AC_H1_EPM,test_AC_H0_SWE,test_proton,test_xray]:
    df['time_tag'] = pd.to_datetime(df['time_tag'].str.slice(0,-1), unit='s')



col_list = ['P1P_.047-.066MEV_IONS_1/(cm**2-s-sr-MeV)',
       'P2P_.066-.114MEV_IONS_1/(cm**2-s-sr-MeV)',
       'P3P_.114-.190MEV_IONS_1/(cm**2-s-sr-MeV)',
       'P4P_.190-.310MEV_IONS_1/(cm**2-s-sr-MeV)',
       'P5P_.310-.580MEV_IONS_1/(cm**2-s-sr-MeV)',
       'P6P_.580-1.05MEV_IONS_1/(cm**2-s-sr-MeV)',
       'P7P_1.05-1.89MEV_IONS_1/(cm**2-s-sr-MeV)',
       'P8P_1.89-4.75MEV_IONS_1/(cm**2-s-sr-MeV)']


# resample and mean
for column in col_list:
    if column=='P1P_.047-.066MEV_IONS_1/(cm**2-s-sr-MeV)':
        lens = train_AC_H1_EPM[column].resample('5T').size().reset_index(name=column+'_lens')
        mean = train_AC_H1_EPM[column].resample('5T').mean().reset_index(name=column+'_mean')
        t = pd.merge(lens,mean,on=['time_tag'],how='left')
    else:
        mean = train_AC_H1_EPM[column].resample('5T').mean().reset_index(name=column+'_mean')
        t = mean.copy()

    if column=='P1P_.047-.066MEV_IONS_1/(cm**2-s-sr-MeV)':
        final_AC_H1_EPM = t
    else:
        final_AC_H1_EPM = pd.merge(final_AC_H1_EPM,t,on=['time_tag'],how='left')

real_train = pd.merge(train_proton,final_AC_H1_EPM,on=['time_tag'],how='left')



train_xray = train_xray.reset_index().set_index('time_tag')

for column in [x for x in train_xray.columns if x!='index']:
    mean = train_xray[column].resample('5T').mean().reset_index(name=column+'_mean')
    t = mean.copy()

    if column=='xs':
        final_xray = t
    else:
        final_xray = pd.merge(final_xray,t,on=['time_tag'],how='left')


real_train = pd.merge(real_train,final_xray,on=['time_tag'],how='left')


col_list = ['H_DENSITY_#/cc','SW_H_SPEED_km/s']
for column in col_list:
    if column=='H_DENSITY_#/cc':
        lens = train_AC_H0_SWE[column].resample('5T').size().reset_index(name=column+'_lens')
        mean = train_AC_H0_SWE[column].resample('5T').mean().reset_index(name=column+'_mean')
        t = pd.merge(lens,mean,on=['time_tag'],how='left')
    else:
        mean = train_AC_H0_SWE[column].resample('5T').mean().reset_index(name=column+'_mean')
        t = mean.copy()

    if column=='H_DENSITY_#/cc':
        final_AC_H0_SWE = t
    else:
        final_AC_H0_SWE = pd.merge(final_AC_H0_SWE,t,on=['time_tag'],how='left')

        
real_train = pd.merge(real_train,final_AC_H0_SWE,on=['time_tag'],how='left')


mustin_columns = [x for x in real_train.columns if x not in ['time_tag','proton']]


# lag variables
lag_list = [1,2,288]

for lag in lag_list:

    real_train = real_train.set_index('time_tag')

    column_list = [x for x in real_train.columns if (x not in ['proton','time']) and ('_lag' not in x)]

    shift_df = real_train[column_list].shift(lag)

    for i in [{x:x+'_lag'+str(lag)} for x in column_list]:
        shift_df = shift_df.rename(columns=i)

    real_train = pd.merge(real_train,shift_df.reset_index(),on='time_tag',how='left')


real_train = real_train.loc[~real_train['proton'].isnull()].reset_index().drop(['index'],axis=1)


time = real_train['time_tag'].dt.time.astype(str)
real_train['time'] = time.str.split(':',expand=True)[0].astype(int)*60+time.str.split(':',expand=True)[1].astype(int)

time = real_test['time_tag'].dt.time.astype(str)
real_test['time'] = time.str.split(':',expand=True)[0].astype(int)*60+time.str.split(':',expand=True)[1].astype(int)


X_train = real_train[[x for x in real_train.columns if x not in ['time_tag','proton']]]
X_test = real_test[[x for x in real_test.columns if x not in ['time_tag','proton']]]

y_train = real_train['proton']


# model train

def rmsle_eval(y_hat, dtrain):
    y = dtrain.get_label()
    
    y = pd.Series(y)
    y = np.where(y<0,0,y)
    
    y_hat = pd.Series(y_hat)
    y_hat = np.where(y_hat<0,0,y_hat)

    return 'myloss', np.mean((np.log(y_hat+1)-np.log(y+1))**2), False


def rmsle(y_hat, dtrain):
    y = dtrain.get_label()
    
    y = pd.Series(y)
    y = np.where(y<0,0,y)
    
    y_hat = pd.Series(y_hat)
    y_hat = np.where(y_hat<0,0,y_hat)
    grad = (1/(y_hat+1))*(np.log(y_hat+1)-np.log(y+1))
    hess = (1/(y_hat+1))**2-(np.log(y_hat+1)-np.log(y+1))/(y_hat+1)**2
    return grad,hess



def get_oof_lgbm(params, train_data, test_data, target_data, num_round, early_round, verbose_round, N_SPLITS=5, random_state=0):

    FOLDs=KFold(n_splits=N_SPLITS, shuffle=True,random_state=0)

    oof = np.zeros(len(train_data))
    predictions = np.zeros(len(test_data))

    feature_importance_df = pd.DataFrame()

    for fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(train_data)):
        trn_data = lgb.Dataset(train_data.iloc[trn_idx], label=target_data.iloc[trn_idx])
        val_data = lgb.Dataset(train_data.iloc[val_idx], label=target_data.iloc[val_idx])

        print("LGB " + str(fold_) + "-" * 50)
        num_round = num_round
        clf = lgb.train(params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=verbose_round,
                        fobj=rmsle,
                        feval = rmsle_eval,
                        early_stopping_rounds = early_round)
        oof[val_idx] = clf.predict(train_data.iloc[val_idx], num_iteration=clf.best_iteration)

        predictions += clf.predict(test_data, num_iteration=clf.best_iteration) / FOLDs.n_splits
        joblib.dump(clf,'lgb'+str(fold_)+'.pkl')
    return oof, predictions, feature_importance_df

import lightgbm as lgb
xgb_params={"objective":"regression",
           "metric":"myloss",
           "max_depth":6,
           "min_child_samples":2,
           "alpha":0.08,
           "gamma":0.06,
           "eta":0.04,
           "subsample":0.08,
           "colsample_bytree":0.97,
           "random_state":2020
           }
a,b,c=get_oof_lgbm(xgb_params, X_train, X_test, y_train, num_round=100000, early_round=400, verbose_round=500, N_SPLITS=5, random_state=0)


test_proton['proton'] = b

