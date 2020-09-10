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



DATASET_PATH = os.path.join('./data/')

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
    args.add_argument("--lr", type=float, default=0.001)
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--num_epochs", type=int, default=100)
    args.add_argument("--print_iter", type=int, default=100)
    args.add_argument("--model_name", type=str, default="model.pth")
    args.add_argument("--prediction_file", type=str, default="./result/prediction.txt")
    args.add_argument("--batch", type=int, default=256)

    config = args.parse_args()
    
    # seed 고정 (reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    base_lr = config.lr
    cuda = config.cuda
    num_epochs = config.num_epochs
    print_iter = config.print_iter
    model_name = config.model_name
    prediction_file = config.prediction_file
    batch = config.batch
    
    # create model
    model = SiameseNetwork()

    if cuda:
        model = model.cuda()

    # define loss function
    class ContrastiveLoss(torch.nn.Module):
        """
        Contrastive loss function.
        Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        """

        def __init__(self, margin=2.0):
            super(ContrastiveLoss, self).__init__()
            self.margin = margin

        def forward(self, output1, output2, label):
            euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
            loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                          (label) * torch.pow(
                torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

            return loss_contrastive

    # get data loader
    # epoch 부분에서 설정 -> 경우의 수 해당 상황에서 sampling
    # can change the number of cpu core (bottle neck)
    tmp_train_dataloader, _ = data_loader(root=DATASET_PATH, phase='train', batch_size=batch)
    # val 총 10,000개라 batch:16 -> 예외없이 다 이용
    validate_dataloader, validate_label_file = data_loader(root=DATASET_PATH, phase='validate', batch_size=16)
    
    num_batches = len(tmp_train_dataloader)

    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=base_lr, eps=1e-8)

    # set scheduler
    # StepLR 경우
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    # warmup + ReduceLROnPlateau
    t_total = len(tmp_train_dataloader) * num_epochs
    warmup_step = int(0.01 * t_total)
    # decay lr, related to a validation
    scheduler_cosine = CosineAnnealingLR(optimizer, t_total)
    scheduler = GradualWarmupScheduler(
        optimizer, 1, warmup_step, after_scheduler=scheduler_cosine
    )


    criterion = ContrastiveLoss()

    #prediction_file = 'prediction.txt'
    counter = []
    loss_history = []
    iteration_number = 0
    best_f1 = 0.0
    best_loss = 1e10

    # check parameter of model
    print("------------------------------------------------------------")
    total_params = sum(p.numel() for p in model.parameters())
    print("num of parameter : ", total_params)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("num of trainable_ parameter :", trainable_params)
    print("------------------------------------------------------------")

    # train
    for epoch in range(0, num_epochs):
        time_ = datetime.datetime.now()
        
        # sample train
        train_dataloader, _ = data_loader(root=DATASET_PATH, phase='train', batch_size=batch)
        for iter_, data in enumerate(train_dataloader, 0):
            iter1_, img0, iter2_, img1, label = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            optimizer.zero_grad()
            model.train()

            output1, output2 = model(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            # cosine scheduler
            scheduler.step()
            if iter_ % print_iter == 0:
                elapsed = datetime.datetime.now() - time_
                expected = elapsed * (num_batches / print_iter)
                _epoch = epoch + ((iter_ + 1) / num_batches)
                print('[{:.3f}/{:d}] loss({}) '
                      'elapsed {} expected per epoch {}'.format(
                    _epoch, num_epochs, loss_contrastive.item(), elapsed, expected))
        # StepLR 경우
        # scheduler.step()
        save_model(str(epoch + 1), model, optimizer)
        if (epoch+1) % 1 == 0:
            # evaluation
            eval_loss = 0.0
            nb_eval_steps = 0

            time_ = datetime.datetime.now()
            for iter_, data in enumerate(validate_dataloader, 0):
                iter1_, img0, iter2_, img1, label = data
                img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
                with torch.no_grad():
                    output1, output2 = model(img0, img1)
                    loss_contrastive = criterion(output1, output2, label)

                    eval_loss += loss_contrastive.item()
                nb_eval_steps += 1
            eval_loss = eval_loss / nb_eval_steps

            elapsed = datetime.datetime.now() - time_
            print('evaluation done^^')
            print('[epoch {}] elapsed: {}'.format(epoch + 1, elapsed))

            if best_loss > eval_loss:
                best_loss = eval_loss
                # save the best model!!!!
                print('achieve best loss!')
                best_dir = os.path.join('./model_save/', 'best_model/')
                os.makedirs(best_dir, exist_ok=True)
                tmp_state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(tmp_state, os.path.join(best_dir, str(epoch + 1) + '.pth'))
                
                
