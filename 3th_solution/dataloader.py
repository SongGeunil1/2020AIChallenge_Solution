# dataset suffle 예정 (조도별, 각도별, 표정별) : Data augmentation
# 조도 -> centercrop, resize, totensor, normalize 로 통일
# 표정별 -> 데이터셋 다양화 ㄱㄱ


import os
import numpy as np
import pandas as pd
import random
from PIL import Image
import torch
from torch.utils import data
from torch.utils.data import Sampler
import torchvision.transforms as transforms


class CustomDataset(data.Dataset):
    def __init__(self, root, phase='train', transform=None):
        self.root = root
        self.phase = phase
        self.labels = {}
        self.transform = transform
        if self.phase != 'train':
            self.label_path = os.path.join(root, self.phase, self.phase + '_label.csv')
            # used to prepare the labels and images path
            self.direc_df = pd.read_csv(self.label_path)
            self.direc_df.columns = ["image1", "image2", "label"]
            self.dir = os.path.join(root, self.phase)
        else:
            self.dir = os.path.join(root, self.phase)
            self.train_meta_dir = os.path.join(root, self.phase, self.phase + '_meta.csv')
            self.train_meta = pd.read_csv(self.train_meta_dir)
            
            # make_true_pair
            self.id_list = list(set(self.train_meta['face_id']))
            
    def __getitem__(self, index):
        if self.phase != 'train':
            # getting the image path
            image0_path = os.path.join(self.dir, self.direc_df.iat[index, 0])
            image1_path = os.path.join(self.dir, self.direc_df.iat[index, 1])
        else:
            # 0.5 확률로 label {0,1} 구성하도록 설정
            # label 0 경우
            if random.random() < 0.5:
                tmp_id = np.random.choice(self.id_list, size=1, replace=False).item()
                candidate = self.train_meta[self.train_meta['face_id'] == int(tmp_id)]
                image0_name = candidate[candidate['cam_angle']=='front'].sample(1)['file_name'].item()
                image0_path = os.path.join(self.dir, image0_name)
                image1_name = candidate[candidate['cam_angle']=='side'].sample(1)['file_name'].item()
                image1_path = os.path.join(self.dir, image1_name)
                image_label = 0
            # label 1 경우
            else:
                tmp_id = np.random.choice(self.id_list, size=1, replace=False).item()
                candidate = self.train_meta[self.train_meta['face_id'] == int(tmp_id)]
                candidate_others = self.train_meta[self.train_meta['face_id'] != int(tmp_id)]
                image0_name = candidate[candidate['cam_angle']=='front'].sample(1)['file_name'].item()
                image0_path = os.path.join(self.dir, image0_name)
                image1_name = candidate_others[candidate_others['cam_angle']=='side'].sample(1)['file_name'].item()
                image1_path = os.path.join(self.dir, image1_name)
                image_label = 1
                
        # Loading the image
        img0 = Image.open(image0_path)
        img1 = Image.open(image1_path)
        # img0 = img0.convert("L")
        # img1 = img1.convert("L")

        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            
        if self.phase == 'validate':
            return (self.direc_df.iat[index, 0], img0, self.direc_df.iat[index, 1], img1,
                    torch.from_numpy(np.array([int(self.direc_df.iat[index, 2])], dtype=np.float32)))
        elif self.phase == 'test':
            dummy = ""
            return (self.direc_df.iat[index, 0], img0, self.direc_df.iat[index, 1], img1, dummy)
        elif self.phase == 'train':
            return (image0_name, img0, image1_name, img1,
                    torch.from_numpy(np.array([int(image_label)], dtype=np.float32)))

    def __len__(self):
        # 원래 train 데이터 980 길이(490명의 2배)    ->     10배로 가정
        return 98000 if self.phase == 'train' else len(self.direc_df)

    def get_label_file(self):
        return "" if self.phase == 'train' else self.label_path

def data_loader(root, phase='train', batch_size=64,):
    if phase == 'train':
        shuffle = True
    else:
        shuffle = False
    dataset = CustomDataset(root, phase,transform=transforms.Compose([transforms.Resize(84),
                                                                      transforms.ToTensor(),
                                                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                                           std=[0.229, 0.224, 0.225])
                                                                      ]))
    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset.get_label_file()


'''
class TrainDataset(data.Dataset):
    def __init__(self, root, phase='train'):
        self.root = root
        self.phase = phase
        
        # data loading (meta data: 전체 데이터 이미지라 봄)
        self.dir = os.path.join(root, self.phase)
        self.train_meta_dir = os.path.join(root, self.phase, self.phase + '_meta.csv')
        self.train_meta = pd.read_csv(self.train_meta_dir)
        
        # 순서 부여(id)
        self.train_meta = self.train_meta.assign(id=self.train_meta.index.values)
        
        # 독립적인 사람 수
        self.id_list = list(set(self.train_meta['face_id']))
        
        # file_name list
        self.file_list = self.train_meta['file_name'].values.tolist()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        file_name = self.file_list[index]
        file_path = os.path.join(self.dir, file_name)
        instance = Image.open(file_path)
        instance = self.transform(instance)
        return instance

    def __len__(self):
        # 66,150
        return len(self.train_meta)
'''


class TrainSampler(Sampler):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 batch_size: int = None):
        """
        # Arguments
            dataset: Instance of torch.utils.data.Dataset from which to draw samples
            batch_size: batch size
        """
        super(TrainSampler, self).__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        
        # 에폭당 iteration 수
        self.num_batch = len(self.dataset) // self.batch_size

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        for _ in range(self.num_batch):
            batch = []
            
            # Sample # of batch size  ->   {0,1} pair로 구성
            episode_classes = np.random.choice(self.dataset.id_list, size=self.batch_size, replace=False).tolist()
            for i, each_id in enumerate(episode_classes):
                # label 0 인 것들
                if i % 2 == 0:
                    candidate = self.dataset.train_meta[self.dataset.train_meta['face_id'] == int(each_id)]
                    batch.append(candidate[candidate['cam_angle']=='front'].sample(1)['id'].item())
                    batch.append(candidate[candidate['cam_angle']=='side'].sample(1)['id'].item())
                # label 1 상황
                else:
                    candidate = self.dataset.train_meta[self.dataset.train_meta['face_id'] == int(each_id)]
                    candidate_others = self.dataset.train_meta[self.dataset.train_meta['face_id'] != int(each_id)]
                    batch.append(candidate[candidate['cam_angle']=='front'].sample(1)['id'].item())
                    batch.append(candidate_others[candidate_others['cam_angle']=='side'].sample(1)['id'].item())

            yield batch
