import gc
import os
import random
import warnings
# from IPython.display import display

import numpy as np
import pandas as pd
from typing import Tuple

import timm
import torch
import optuna
import torch.nn as nn  
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# 导入 PyTorch 的 DataParallel 模块
from torch.nn.parallel import DataParallel # 单机多卡的分布式训练（数据并行） 模型训练加速

from datetime import datetime

from collections import OrderedDict

warnings.filterwarnings('ignore', category=Warning)
gc.collect()


labels = ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other']

class Config:
    seed = 3131
    image_transform = transforms.Resize((512,512))
    batch_size = 100
    num_epochs = 9
    num_folds = 5
    num_trials = 20
    dataset_wide_mean = 0
    dataset_wide_std = 0
    manual_pruning_threshold = 0.57
    optimize_hyperparameters = False
    
class HyperparameterSpaces:
    lowpass = {
        "min": np.exp(10),
        "max": np.exp(10)
    }
    highpass = {
        "min": np.exp(-6),
        "max": np.exp(-6)
    }
    learning_rate = {
        "min": 0.0005,
        "max": 0.0015
    }
    dropout = {
        "min": 0.17,
        "max": 0.23
    }

    schedulers = ["CosineAnnealingLR", "ReduceLROnPlateau"]
    normalize_dataset_wide = [True, False]

class HyperparameterPreset:
    lowpass = np.exp(10)
    highpass = np.exp(-6)
    learning_rate = 0.00137263241151172
    dropout = 0.184235721122803
    scheduler = "CosineAnnealingLR"
    normalize_dataset_wide = True
    
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def kl_loss(p, q):
    epsilon = 10 ** (-15)
    
    p = torch.clamp(p, epsilon, 1 - epsilon)
    log_p = torch.log(p)
    log_q = nn.functional.log_softmax(q, dim=1)
    
    kl_divergence_per_point = p * (log_p - log_q)
    kl_divergence_per_label = torch.sum(kl_divergence_per_point, dim=1)
    
    return torch.mean(kl_divergence_per_label)

set_seed(Config.seed)
gc.collect()



train_df = pd.read_csv("/home/xyli/kaggle/train.csv")

def extract_vote_count_features(input_data: pd.DataFrame) -> pd.DataFrame:
    label_votes = pd.DataFrame()
    
    for label in labels:
        input_grouped_by_spectrogram_id = input_data[f'{label}_vote'].groupby(input_data['spectrogram_id']).sum()

        label_vote_sum = pd.DataFrame()
        label_vote_sum["spectrogram_id"] = input_grouped_by_spectrogram_id.index
        label_vote_sum[f"{label}_vote_sum"] = input_grouped_by_spectrogram_id.values

        if label == labels[0]:
            label_votes = label_vote_sum
        else:
            label_votes = label_votes.merge(label_vote_sum, on='spectrogram_id', how='left')
            
    return label_votes

def extract_features(input_data: pd.DataFrame) -> pd.DataFrame:
    choose_cols = ['spectrogram_id']
    feature_df = extract_vote_count_features(input_data)
    
    feature_df['total_vote'] = 0
    for label in labels:
        choose_cols += [f'{label}_vote']
        feature_df['total_vote'] += feature_df[f'{label}_vote_sum']
        
    for label in labels:
        feature_df[f'{label}_vote'] = feature_df[f'{label}_vote_sum'] / feature_df['total_vote']
        
    feature_df = feature_df[choose_cols]
    feature_df['path'] = feature_df['spectrogram_id'].apply(lambda x: "/home/xyli/kaggle/train_spectrograms/" + str(x) + ".parquet")
    
    return feature_df

train_features = extract_features(train_df)
# display(train_features)
    
gc.collect()

def preprocess(path_to_parquet, lowpass, highpass):
    data = pd.read_parquet(path_to_parquet)
    data = data.fillna(-1).values[:, 1:].T
    data = np.clip(data, highpass, lowpass)
    data = np.log(data)
    
    return data

def get_dataset_wide_mean(paths, lowpass, highpass):
    data_sum = 0
    num_values = 0

    for path in paths:      
        data_point = preprocess(path[0], lowpass, highpass)
        data_sum += data_point.sum(axis=(0, 1))
        rows, columns = data_point.shape
        num_values += rows * columns
    
    return data_sum / num_values

def get_dataset_wide_std(paths, lowpass, highpass):
    sum_of_stds = 0
    num_values = 0
    
    for path in paths:
        data_point = preprocess(path[0], lowpass, highpass)
        sum_of_stds += np.sum((data_point - Config.dataset_wide_mean) ** 2)
        rows, columns = data_point.shape
        num_values += rows * columns
    
    return np.sqrt(sum_of_stds / (num_values - 1))

def normalize_dataset_wide(data_point):
    eps = 1e-6

    data_point = (data_point - Config.dataset_wide_mean) / (Config.dataset_wide_std + eps)

    data_tensor = torch.unsqueeze(torch.Tensor(data_point), dim=0)
    data_point = Config.image_transform(data_tensor)

    return data_point

def normalize_instance_wise(data_point):
    eps = 1e-6
    
    data_mean = data_point.mean(axis=(0, 1))
    data_std = data_point.std(axis=(0, 1))
    data_point = (data_point - data_mean) / (data_std + eps)
    
    data_tensor = torch.unsqueeze(torch.Tensor(data_point), dim=0)
    data_point = Config.image_transform(data_tensor)
    
    return data_point

def get_batch(paths, lowpass, highpass, normalization_dataset_wide):        
    batch_data = []
    
    for path in paths:
        data_point = preprocess(path[0], lowpass, highpass)
        
        if normalization_dataset_wide:
            data_point = normalize_dataset_wide(data_point)
        else:
            data_point = normalize_instance_wise(data_point)
        
        batch_data.append(data_point)
    batch_data = torch.stack(batch_data)

    return batch_data


# print("Calculating dataset-wide mean...")
# Config.dataset_wide_mean = get_dataset_wide_mean(train_features[["path"]].values, HyperparameterPreset.lowpass, HyperparameterPreset.highpass)
# print("Finished mean calculation!\nCalculating dataset-wide standard deviation...")
# Config.dataset_wide_std = get_dataset_wide_std(train_features[["path"]].values, HyperparameterPreset.lowpass, HyperparameterPreset.highpass)
# print("Finished standard deviation calculation!")



gc.collect()


"""
Config.dataset_wide_mean:  -0.2972692229201065
Config.dataset_wide_std:  2.5997336315611026
"""

Config.dataset_wide_mean = -0.2972692229201065
Config.dataset_wide_std = 2.5997336315611026
print('Config.dataset_wide_mean: ', Config.dataset_wide_mean)
print('Config.dataset_wide_std: ', Config.dataset_wide_std)



# size是图片的shape，即（b,c,w,h）
# lam 参数的作用是调整融合区域的大小，lam 越接近 1，融合框越小，融合程度越低
# 返回一个随机生成的矩形框，用于确定两张图像的融合区域
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# data:(b,c,w,h)
# targets1~3:(b,),说明每张图中有3类目标
# alpha,两张图片融合区域的大小
# 返回这b个图随机两两融合的结果data:(b,c,w,h),和融合结果的标签
# 我感觉融合结果的标签类别1应该是(targets1 + shuffled_targets1),类别2、3同理
def cutmix(data, targets1, alpha):
    # 对b这个维度进行随机打乱,产生随机序列indices
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices] # 这是打乱b后的数据,shape=(b,c,w,h)
    shuffled_targets1 = targets1[indices] # 同上shape=(b,)

    
    # 基于 alpha 随机生成 lambda 值，它控制了两个图像的融合程度
    lam = np.random.beta(alpha, alpha)
    
    # 随机生成一个矩形框 (bbx1, bby1) 和 (bbx2, bby2)，用于融合两张图像的区域
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    
    # 使用另一张图像的相应区域替换第一张图像的相应区域，实现图像融合
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    
    # adjust lambda to exactly match pixel ratio
    # λ = 1 - (融合区域的像素数量 / 总图像像素数量)
    # 基于现实对已给的λ进行一个调整
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    targets = [targets1, shuffled_targets1, lam]
    return data, targets

# 我感觉输入参数和输出参数含义同上
def mixup(data, targets1, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]

    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam) # 我感觉是对于每个像素点都做该算法
    targets = [targets1, shuffled_targets1, lam]

    return data, targets

# 被我猜中了，这里是1张图3个目标，因为经过cutmix，每个目标变得半人半马，于是求每个目标loss时需要用到2个标签，最后该图片的总loss是3个目标的loss和
# preds1~3是预测出图中3个目标，对其softmax的概率向量
# targets中有6个标签，其中2个为1组，代表一个目标的类别，最后一个是lam，代表cutmix的程度，求loss时可做半人半马的权重
def cutmix_criterion(preds1, targets):
    targets1, targets2, lam = targets[0], targets[1], targets[2]
#     criterion = nn.CrossEntropyLoss(reduction='mean').cuda()
    return lam * kl_loss(targets1 ,preds1) + (1 - lam) * kl_loss(targets2 ,preds1)

# 同上
def mixup_criterion(preds1, targets):
    targets1, targets2, lam = targets[0], targets[1], targets[2]
#     criterion = nn.CrossEntropyLoss(reduction='mean')
    return lam * kl_loss(targets1 ,preds1) + (1 - lam) * kl_loss(targets2 ,preds1)


def get_fold_train_val_indexes(indexes: np.ndarray, fold: int) -> Tuple[np.ndarray, np.ndarray]:
    lower_bound = fold * len(indexes) // Config.num_folds
    upper_bound = (fold + 1) * len(indexes) // Config.num_folds
    
    val_idx = indexes[lower_bound:upper_bound]
    train_idx = []
    
    for index in indexes:
        if index not in val_idx:
            train_idx.append(index)
            
    train_idx = np.array(train_idx)
    
    return (train_idx, val_idx) 

def objective(trial) -> float:    
    if trial is None:
        lowpass = HyperparameterPreset.lowpass
        highpass = HyperparameterPreset.highpass
        learning_rate = HyperparameterPreset.learning_rate
        dropout = HyperparameterPreset.dropout
        scheduler_name = HyperparameterPreset.scheduler
        normalize_dataset_wide = HyperparameterPreset.normalize_dataset_wide
    else:
        lowpass = trial.suggest_float("lowpass", HyperparameterSpaces.lowpass["min"], HyperparameterSpaces.lowpass["max"])
        highpass = trial.suggest_float("highpass", HyperparameterSpaces.highpass["min"], HyperparameterSpaces.highpass["max"])
        learning_rate = trial.suggest_float("learning_rate", HyperparameterSpaces.learning_rate["min"], HyperparameterSpaces.learning_rate["max"])
        dropout = trial.suggest_float("dropout", HyperparameterSpaces.dropout["min"], HyperparameterSpaces.dropout["max"])
        scheduler_name = trial.suggest_categorical("scheduler", HyperparameterSpaces.schedulers)
        normalize_dataset_wide = trial.suggest_categorical("normalize_dataset_wide", HyperparameterSpaces.normalize_dataset_wide)
    
    for fold in range(Config.num_folds):        
        train_idx, val_idx = get_fold_train_val_indexes(train_spectrogram_indexes, fold)

        model = timm.create_model(
            'efficientnet_b1', 
            pretrained=False, 
            num_classes=6, 
            in_chans=1, 
            drop_rate=dropout
        ).to(device)
        

        if fold!=0:
            model.load_state_dict(torch.load(f'/home/xyli/kaggle/efficientnet_b1_fold{fold}.pth'))
        else:
            state_dict = torch.load(f'/home/xyli/kaggle/Kaggle_HMS/efficientnet_b1_fold{fold}.pth')  # 模型pth文件
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # 去掉前缀（去掉前七个字符）
                new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
            model.load_state_dict(new_state_dict, strict=True)  # 重新加载这个模型
            # ckpt = torch.load(f'/home/xyli/kaggle/Kaggle_HMS/efficientnet_b1_fold{fold}.pth')
            # model.load_state_dict(ckpt, strict=False)


        model = DataParallel(model)
        
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            betas=(0.5, 0.999),
            weight_decay=0.01
        )
            
        if scheduler_name == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(optimizer, T_max=Config.num_epochs)
        elif scheduler_name == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(optimizer)
        else:
            raise ValueError()

        best_val_loss = float('inf')
        train_losses = []
        val_losses = []

        print(f"Starting training for fold {fold + 1}")

        for epoch in range(Config.num_epochs):
            print(f" Epoch: {epoch + 1}")
            model.train()
            train_loss = []

            random_num = np.arange(len(train_idx))
            np.random.shuffle(random_num)
            train_idx = train_idx[random_num]

            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  Train - {len(train_idx)} indexes")
            for idx in range(0, len(train_idx), Config.batch_size):
                optimizer.zero_grad()

                train_batch_idx = train_idx[idx:idx + Config.batch_size]
                train_batch_idx_paths = train_features[['path']].iloc[train_batch_idx].values
                train_batch = get_batch(train_batch_idx_paths, lowpass, highpass, normalize_dataset_wide)
                train_batch = train_batch.to(device)

                train_target = train_features[['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']].iloc[train_batch_idx].values
                train_target = torch.Tensor(train_target).to(device)
                
            
                """
                train_batch:  torch.Size([16, 1, 512, 512])
                train_target:  torch.Size([16, 6])
                train_pred:  torch.Size([16, 6])
                """
                
                
                target = train_target
                input = train_batch
                random_number = random.random() # 生成一个0到1之间的随机数
                if random_number < 0.3:
                    input,targets=cutmix(input,target,0.2)

                    targets[0]=torch.tensor(targets[0]).cuda()
                    targets[1]=torch.tensor(targets[1]).cuda()
                    targets[2]=torch.tensor(targets[2]).cuda()
                elif random_number > 0.7:
                    input,targets=mixup(input,target,0.2)

                    targets[0]=torch.tensor(targets[0]).cuda()
                    targets[1]=torch.tensor(targets[1]).cuda()
                    targets[2]=torch.tensor(targets[2]).cuda()
                else:
                    None
                
                
                input = input.cuda()
                target = target.cuda()
                
                output = model(input)
                
                loss=None
                if random_number < 0.3:
                    loss = cutmix_criterion(output, targets) # 注意这是在CPU上运算的
                elif random_number > 0.7:
                    loss = mixup_criterion(output, targets) # 注意这是在CPU上运算的
                else:
                    loss = kl_loss(target, output)

                
                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())

                print(f'epoch:{epoch}, idx:{idx:6d}, loss:{loss:.6f}') 

                with open("log.txt", "a") as f:
                    print(f'epoch:{epoch}, idx:{idx:6d}, loss:{loss:.6f}', file=f)


            epoch_train_loss = np.mean(train_loss)
            train_losses.append(epoch_train_loss)
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Epoch {epoch + 1}: Train Loss = {epoch_train_loss:.2f}")
            with open("log.txt", "a") as f:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Epoch {epoch + 1}: Train Loss = {epoch_train_loss:.2f}", file=f)


            if scheduler_name == "CosineAnnealingLR":
                scheduler.step()

            model.eval()
            val_loss = []

            with torch.no_grad():
                print(f"  Validation - {len(val_idx)} indexes")
                for idx in range(0, len(val_idx), Config.batch_size):
                    val_batch_idx = val_idx[idx:idx + Config.batch_size]
                    val_batch_idx_paths = train_features[['path']].iloc[val_batch_idx].values
                    val_batch = get_batch(val_batch_idx_paths, lowpass, highpass, normalize_dataset_wide)
                    val_batch = val_batch.to(device)

                    val_target = train_features[['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']].iloc[val_batch_idx].values
                    val_target = torch.Tensor(val_target).to(device)

                    val_pred = model(val_batch)

                    loss = kl_loss(val_target, val_pred)
                    val_loss.append(loss.item())

            epoch_val_loss = np.mean(val_loss)
            val_losses.append(epoch_val_loss)
            print(f" Epoch {epoch + 1}: Test Loss = {epoch_val_loss:.2f}")
            with open("log.txt", "a") as f:
                print(f" Epoch {epoch + 1}: Test Loss = {epoch_val_loss:.2f}", file=f)

            
            if scheduler_name == "ReduceLROnPlateau":
                    scheduler.step(epoch_val_loss)

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                torch.save(model.state_dict(), f"efficientnet_b1_fold{fold}.pth")

            gc.collect()
            
            if trial is not None:
                trial.report(epoch_val_loss, epoch)

                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        print(f"Fold {fold + 1} Best Test Loss: {best_val_loss:.2f}")
        with open("log.txt", "a") as f:
            print(f"Fold {fold + 1} Best Test Loss: {best_val_loss:.2f}", file=f)

    
    return best_val_loss

train_spectrogram_indexes = np.arange(len(train_features))
np.random.shuffle(train_spectrogram_indexes)
    
gc.collect()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

gc.collect()

if Config.optimize_hyperparameters:
    Config.num_folds = 2
    Config.num_epochs = 2
    
    print("### STARTING HYPERPARAMETER OPTIMIZATION ###")
    study = optuna.create_study(pruner=optuna.pruners.ThresholdPruner(upper=Config.manual_pruning_threshold))
    study.optimize(objective, n_trials=Config.num_trials)

    print("Best test loss:", study.best_value)
    print("Best trial run:", study.best_trial)
    print("Hyperparameter values for best test loss:")
    print(study.best_params)
    print("### FINISHED HYPERPARAMETER OPTIMIZATION ###")
else:
    print("### STARTING MODEL TRAINING ###")
    objective(None)
    print("### FINISHED MODEL TRAINING ###")