import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as trans
from lib.config import get_argparser
from lib.datasets.sub3 import  STAGE_dataset 
from lib.loss import OrdinalRegressionLoss
from lib.model.networks import Basemodel
import warnings
warnings.filterwarnings('ignore')
import random
from tqdm import tqdm
from sklearn.metrics import f1_score


def get_dataloader(args):
    filelists = os.listdir(args.train_root)
    train_filelists, val_filelists = train_test_split(filelists, test_size=args.val_ratio, random_state=42)

    print("Total Nums: {}, train: {}, val: {}".format(len(filelists), len(train_filelists), len(val_filelists)))

    for id in train_filelists:
        f = open('trainval_split.txt', 'a', encoding='utf-8')
        f.write(id + '\n')
        f.close()
    
    for id in val_filelists:
        f = open('trainval_split.txt', 'a', encoding='utf-8')
        f.write(id + '\n')
        f.close()

   
    oct_train_transforms = trans.Compose([
        trans.ToTensor(),
        # trans.RandomHorizontalFlip(),
        # trans.RandomVerticalFlip()
    ])

    oct_val_transforms = trans.Compose([
        # trans.CenterCrop(512),
        trans.ToTensor(),
    ])

    train_dataset =  STAGE_dataset(
                        dataset_root=args.train_root, 
                        aux_info_file=args.aux_info_file,
                        oct_transforms=oct_train_transforms,
                        filelists=train_filelists,
                        label_file=args.label_file,
                        )

    val_dataset =  STAGE_dataset(
                        dataset_root=args.train_root, 
                        oct_transforms=oct_val_transforms,
                        filelists=val_filelists,
                        aux_info_file=args.aux_info_file,
                        label_file=args.label_file)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bs,
        shuffle=False
    )

    return train_loader, val_loader

def train(args, model, iters, train_loader, val_dataloader, optimizer, criterion, log_interval, eval_interval):
    time_start = time.time()
    iter = 0

    if args.model_mode == "18" or args.model_mode == "34" or args.model_mode == 'x50':
        log_file = './logs/resnet' + args.model_mode + '_' + time.strftime(
            '%m-%d-%H-%M',
            time.localtime(
                time.time())) + '.txt'
        model_path = './models/resnet' + args.model_mode 

    if os.path.exists(model_path) == False:
        os.makedirs(model_path)

    model.train()
    model = model.cuda()
    criterion = criterion.cuda()
    if os.path.exists(model_path) == False:
        os.makedirs(model_path)

    avg_loss_list = []
    avg_kappa_list = []
    best_f1 = 0.
    while iter < iters:
        # for data in train_dataloader:
        for oct_imgs, info_ids, labels in tqdm(train_loader):
            iter += 1
            if iter > iters:
                break

            optimizer.zero_grad()

            info_ids = info_ids.cuda()  # 同理
            oct_imgs = oct_imgs.cuda()  # 同理
            labels = labels.reshape(-1, 1).cuda()  # 同理

            logits = model(oct_imgs, info_ids)
            loss, like_hoods = criterion(logits.reshape(-1,1), labels)
            # acc = paddle.metric.accuracy(input=logits, label=labels.reshape((-1, 1)), k=1)
            for p, l in zip(like_hoods.detach().cpu().numpy().argmax(1), labels.view(-1).detach().cpu().numpy()):
                avg_kappa_list.append([p, l])
            
            loss.backward()
            optimizer.step()
         
            avg_loss_list.append(loss.detach().cpu().numpy())  ##.detach().cpu().numpy()[0]

            if iter % log_interval == 0:
                avg_loss = np.array(avg_loss_list).mean()
                avg_kappa_list = np.array(avg_kappa_list)

                # avg_kappa = cohen_kappa_score(avg_kappa_list[:, 0], avg_kappa_list[:, 1], weights='quadratic')
                pred = avg_kappa_list[:, 0]
                gt = avg_kappa_list[:, 1]

                try:
                    acc = np.sum(pred == gt) / len(pred)
                    ma_f1 = f1_score(pred, gt, labels=[0,1,2,3,4], average='macro')
                    mi_f1 = f1_score(pred, gt, labels=[0,1,2,3,4], average='micro')
                    
                except:
                    acc = 0


                avg_loss_list = []
                avg_kappa_list = []
                f = open(log_file, 'a', encoding='utf-8')
                f.write(
                    "[TRAIN] iter={}/{} avg_loss={:.4f} ma_f1={:.4f} mi_f1={:.4f} acc={:.4f}".format(
                        iter, iters, avg_loss, ma_f1, mi_f1, acc))
                f.write('\n')
                f.close()
                print(
                    "[TRAIN] iter={}/{} avg_loss={:.4f} ma_f1={:.4f} mi_f1={:.4f} acc={:.4f}".format(
                        iter, iters, avg_loss, ma_f1, mi_f1, acc))
                print('time cost: {:2.2f}s'.format(time.time() - time_start))
                time_start = time.time()
                f1_train = (ma_f1 + mi_f1) /2 

            if iter % eval_interval == 0:
                avg_loss, ma_f1, mi_f1, acc = val(model, val_dataloader, criterion)
                f = open(log_file, 'a', encoding='utf-8')
                f.write(
                    "[EVAL] iter={}/{} avg_loss={:.4f} ma_f1={:.4f} mi_f1={:.4f} acc={:.4f}".format(
                        iter, iters, avg_loss, ma_f1, mi_f1, acc))
                f.write('\n')
                f.close()
                print(
                    "[EVAL] iter={}/{} avg_loss={:.4f} ma_f1={:.4f} mi_f1={:.4f} acc={:.4f}".format(
                        iter, iters, avg_loss, ma_f1, mi_f1, acc))
                
                f1_val = (ma_f1 + mi_f1) /2 
                if f1_val >= best_f1:
                    best_f1 = f1_val
                    torch.save(model,
                               os.path.join(model_path,
                                            "best_model_{:.4f}.pth".format(best_f1)))  ###torch1.6
            model.train()


def val(model, var_loader, criterion):
    model.eval()
    avg_loss_list = []
    cache = []
    with torch.no_grad():
        # for data in val_dataloader:
        for oct_imgs, info_ids, labels in var_loader:
            info_ids = info_ids.cuda()  # 同理
            oct_imgs = oct_imgs.cuda()  # 同理
            labels = labels.reshape(-1, 1).cuda()  # 同理
            
            logits = model(oct_imgs, info_ids)
            loss, like_hoods = criterion(logits.reshape(-1,1), labels)
            
            for p, l in zip(like_hoods.detach().cpu().numpy().argmax(1), labels.view(-1).detach().cpu().numpy()):
                cache.append([p, l])
            avg_loss_list.append(loss.detach().cpu().numpy())
            
    cache = np.array(cache)
    kappa = cohen_kappa_score(cache[:, 0], cache[:, 1], weights='quadratic')
    avg_loss = np.array(avg_loss_list).mean()

    pred = cache[:, 0]
    gt = cache[:, 1]
    try:
        acc = np.sum(pred == gt) / len(pred)
        ma_f1 = f1_score(pred, gt, labels=[0,1,2,3,4], average='macro')
        mi_f1 = f1_score(pred, gt, labels=[0,1,2,3,4], average='micro')
    except:
        acc = 0

    return avg_loss, ma_f1, mi_f1, acc

def main():
    args = get_argparser().parse_args()
    train_loader, val_dataloader = get_dataloader(args)
    model = Basemodel(args.model_mode)
    
    if args.optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.loss_type == 'OL':
        criterion = OrdinalRegressionLoss(num_class=5)
    elif args.loss_type == 'ce':
        criterion = torch.nn.CrossEntropyLoss()

    train(args, model, args.iters, train_loader, val_dataloader, optimizer, criterion, 10, 10)



if __name__ == '__main__':
    # root = 'data/STAGE_training/training_images'
    # sub_list = os.listdir(root)
    # for i in sub_list:
    #     path = os.path.join(root, i, '.DS_Store')
    #     try:
    #         os.remove(path)
    #         print(path)
    #     except:
    #         pass
    main()