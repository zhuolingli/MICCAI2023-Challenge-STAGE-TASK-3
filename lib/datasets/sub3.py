from torch.utils.data import DataLoader,Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
import torchvision.transforms as trans

from PIL import Image

# AUX_INFO_DICT = {3:['normal', 'early', 'intermediate', 'advanced'],
#                  0:['male', 'female'],
#                  1:[],
#                  2:['OD','OS']}

AUX_INFO = ['male', 'female'] + [] + ['OD','OS'] + ['normal', 'early', 'intermediate', 'advanced'] 

AUX_INFO_DICT = {value:id for id, value in enumerate(AUX_INFO)}
    
    

class STAGE_dataset(Dataset):  #### oct_image[150:662,:]
  
    def __init__(self,
                 oct_transforms,
                 dataset_root,
                 aux_info_file='',
                 label_file='',
                 filelists=None,
                 num_classes=5,
                 mode='train'):

        self.dataset_root = dataset_root
        self.oct_transforms = oct_transforms
        self.mode = mode.lower()
        self.num_classes = num_classes
        self.aux_info_file = aux_info_file
        aux_info = {row['ID']: row[1:].values
                     for _, row in pd.read_excel(self.aux_info_file).iterrows()}
        if self.mode == 'train':
            label = {row['ID']: row[1:].values
                     for _, row in pd.read_excel(label_file).iterrows()}

            self.file_list = [[f, aux_info[int(f)], label[int(f)]] for f in os.listdir(dataset_root)]
            
        elif self.mode == "test":
            self.file_list = [[f, aux_info[int(f)], None] for f in os.listdir(dataset_root)]

        if filelists is not None:
            self.file_list = [item for item in self.file_list if item[0] in filelists]

    def __getitem__(self, idx):
        real_index, aux_info, label = self.file_list[idx]

        oct_series_list = sorted(os.listdir(os.path.join(self.dataset_root, real_index)),
                                 key=lambda x: int(x.split("_")[0]))

        oct_series_0 = cv2.imread(os.path.join(self.dataset_root, real_index, oct_series_list[0]),
                                  cv2.IMREAD_GRAYSCALE)

        oct_img = np.zeros((len(oct_series_list), oct_series_0.shape[1], oct_series_0.shape[1]), dtype="uint8") # resize to 512x512

        for k, p in enumerate(oct_series_list):
            oct_img[k] = cv2.imread(
                os.path.join(self.dataset_root, real_index, p), cv2.IMREAD_GRAYSCALE)[150:662,:] # 只取了150-662

        oct_img = oct_img.transpose(2, 1, 0) ####ljc

        info_id = []
        for idx, value in enumerate(aux_info):
            if idx == 1:
                continue
            info_id.append(AUX_INFO_DICT[value])
        info_id = np.array(info_id)
        if self.oct_transforms is not None:
            oct_img = self.oct_transforms(oct_img)

        if self.mode == 'test':
            return oct_img, info_id, real_index
        if self.mode == "train":
            return oct_img, info_id, label

    def __len__(self):
        return len(self.file_list)

