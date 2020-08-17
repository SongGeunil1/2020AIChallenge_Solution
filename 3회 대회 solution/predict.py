import os
import random
import math
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
import argparse
from torch import optim
from torch.autograd import Variable
import torchvision.utils

from dataloader import data_loader
from evaluation import evaluation_metrics
from model import SiameseNetwork
from scheduler import GradualWarmupScheduler
import torch.nn.functional as F

'''
!!!!!!!!!!!!!!!!!!!!! 필독!!!!!!!!!!!!!!!!!!!!!!!!!!!
** 컨테이너 내 기본 제공 폴더
- /datasets : read only 폴더 (각 태스크를 위한 데이터셋 제공)
- /tf/notebooks :  read/write 폴더 (참가자가 Wirte 용도로 사용할 폴더)
1. 참가자는 /datasets 폴더에 주어진 데이터셋을 적절한 폴더(/tf/notebooks) 내에 복사/압축해제 등을 진행한 뒤 사용해야합니다.
   예시> Jpyter Notebook 환경에서 압축 해제 예시 : !bash -c "unzip /datasets/objstrgzip/18_NLP_comments.zip -d /tf/notebooks/
   예시> Terminal(Vs Code) 환경에서 압축 해제 예시 : bash -c "unzip /datasets/objstrgzip/18_NLP_comments.zip -d /tf/notebooks/
   
2. 참가자는 각 문제별로 데이터를 로드하기 위해 적절한 path를 코드에 입력해야합니다. (main.py 참조)
3. 참가자는 모델의 결과 파일(Ex> prediction.txt)을 write가 가능한 폴더에 저장되도록 적절 한 path를 입력해야합니다. (main.py 참조)
4. 세션/컨테이너 등 재시작시 위에 명시된 폴더(datasets, notebooks) 외에는 삭제될 수 있으니 
   참가자는 적절한 폴더에 Dataset, Source code, 결과 파일 등을 저장한 뒤 활용해야합니다.
   
!!!!!!!!!!!!!!!!!!!!! 필독!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''

DATASET_PATH = os.path.join('./data/')

def _infer(model, cuda, data_loader):
    res_id = []
    res_fc = []

    euclidean_distances = []
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            iter1_, x0, iter2_, x1, label = data
            if cuda:
                x0 = x0.cuda()
                x1 = x1.cuda()
            output1, output2 = model(x0, x1)
            euclidean_distance = F.pairwise_distance(output1, output2).cpu()
            euclidean_distances.append(euclidean_distance)

    temp = sorted(euclidean_distances)[int(len(euclidean_distances) / 2)]
    for index, data in enumerate(data_loader):
        iter1_, x0, iter2_, x1, label = data
        image_name = iter1_[0] + ' ' + iter2_[0]
        if euclidean_distances[index] < temp:
            result = 0
        else:
            result = 1
        res_fc.append(result)
        res_id.append(image_name)
    return [res_id, res_fc]

def feed_infer(output_file, infer_func):
    prediction_name, prediction_class = infer_func()

    print('write output')
    predictions_str = []
    for index, name in enumerate(prediction_name):
        test_str = name + ' ' + str(prediction_class[index])
        predictions_str.append(test_str)
    with open(output_file, 'w') as file_writer:
        file_writer.write("\n".join(predictions_str))

    if os.stat(output_file).st_size == 0:
        raise AssertionError('output result of inference is nothing')

def test(prediction_file, model, test_dataloader, cuda):
    feed_infer(prediction_file, lambda : _infer(model, cuda, data_loader=test_dataloader))


def save_model(model_name, model, optimizer):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        # 'scheduler': scheduler.state_dict()
    }
    torch.save(state, os.path.join('./model_save/', model_name + '.pth'))
    print('model saved')


def load_model(model_name, model, optimizer=None, scheduler=None):
    state = torch.load(os.path.join('./model_save/', model_name))
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    print('model loaded')



if __name__ == '__main__':
    # mode argument
    args = argparse.ArgumentParser()
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--model_name", type=str, default="14.pth")
    args.add_argument("--prediction_file", type=str, default="./result/prediction.txt")

    config = args.parse_args()
    
    # seed 고정 (reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    cuda = config.cuda
    model_name = config.model_name
    prediction_file = config.prediction_file
    
    # create model
    model = SiameseNetwork()
    
    # load model
    load_model(model_name, model)

    if cuda:
        model = model.cuda()

    model.eval()
    
    # accuracy 확인
    test_dataloader, _ = data_loader(root=DATASET_PATH, phase='test', batch_size=1)
    
    #prediction_file = "prediction_test.txt"
    test(prediction_file, model, test_dataloader, cuda)