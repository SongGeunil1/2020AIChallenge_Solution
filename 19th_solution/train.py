import logging
import os
import numpy as np
from tqdm import tqdm, trange
import random
import lightgbm as lgb
from sklearn.linear_model import Ridge
import joblib
import shutil
from aifactory.modules import activate, submit
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


def evaluate(y_hat, dtrain):
    # rmsle 구현
    y = dtrain.get_label()
    
    y = pd.Series(y)
    y = np.where(y<0,0,y)
    
    y_hat = pd.Series(y_hat)
    y_hat = np.where(y_hat<0,0,y_hat)

    return 'myloss', np.mean((np.log(y_hat+1)-np.log(y+1))**2), False

def evaluate_train(y_hat, dtrain):
    # rmsle의 미분 함수 구현 --> 학습에 활용
    y = dtrain.get_label()
    
    y = pd.Series(y)
    y = np.where(y<0,0,y)
    
    y_hat = pd.Series(y_hat)
    y_hat = np.where(y_hat<0,0,y_hat)
    grad = (1/(y_hat+1))*(np.log(y_hat+1)-np.log(y+1))
    hess = (1/(y_hat+1))**2-(np.log(y_hat+1)-np.log(y+1))/(y_hat+1)**2

    return grad,hess



def train_lgbm(params, train_data, test_data, target_data, num_round, early_round, verbose_round, N_SPLITS=5, random_state=0):

    FOLDs=KFold(n_splits=N_SPLITS, shuffle=True,random_state=0)

    oof = np.zeros(len(train_data))
    predictions = np.zeros(len(test_data))
    feature_importance_df = pd.DataFrame()
    iter_list=[]

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
        iter_list.append(clf.best_iteration)
        predictions += clf.predict(test_data, num_iteration=clf.best_iteration) / FOLDs.n_splits
        joblib.dump(clf,'/tf/notebooks/19th_problem/lightgbm_model/lgb'+str(fold_)+'.pkl')
    return oof, predictions, np.mean(iter_list)
    

    
def main():
    args = argparse.ArgumentParser()
    args.add_argument("--input_max_len", default=400, type=int,
                            help="Maximum sequence length for audio")
    args.add_argument("--num_epochs", default=300, type=int,
                            help="num_epochs")
    args.add_argument("--data_path", default='data', type=str,
                            help="root")
    args.add_argument("--sample_rate", default=16000, type=int,
                            help="sampling rate for audio")
    args.add_argument("--n_fft_size", default=400, type=int,
                            help="time widnow for fourier transform")
    args.add_argument("--n_mfcc", default=40, type=int,
                            help="low frequency range (from 0 to n_mfcc)")
    args.add_argument("--max_len", default=30, type=int,
                            help="target_max_length")
    args.add_argument("--batch_size", default=128, type=int,
                            help="target_max_length")
    args.add_argument(
        "--warmup_percent", default=0.1, type=float, help="Linear warmup over warmup_percent."
    )
    args.add_argument(
        "--when", type=int, default=5, help="when to decay learning rate (default: 20)"
    )
    args.add_argument(
        "--clip", type=float, default=0.8, help="gradient clip value (default: 0.8)"
    )
    args.add_argument(
        "--lr", type=float, default=1e-4, help="initial learning rate (default: 1e-3)"
    )
    args.add_argument("--seed", type=int, default=1234, help="random seed")
    args.add_argument(
        "--logging_steps", type=int, default=50, help="frequency of result logging (default: 30)"
    )
    config = args.parse_args()
    set_seed(config)


    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO 

        )


    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device

    
    train_lgbm(model, train_loader, validate_loader, loss_fct, config, vocab)
    logger.info('Done Training')
if __name__ =='__main__':
    main()