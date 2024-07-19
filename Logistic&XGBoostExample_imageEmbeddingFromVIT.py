# for demonstration purpose, interaction with amazon S3, conducting data analysis, Extract embeddings from ViT, pass them through ML model to do classification , contents related to actual analysis are removed, generic code that can be adapted to any conditions
# install packages
!pip install timm
!pip install einops
!pip install wandb
!pip install nibabel
!pip install nilearn
!pip install seaborn
!pip install fsspec==2023.6.0
!pip install s3fs==2023.6.0
!pip install boto3==1.34.51
!pip install botocore==1.34.51

!pip install monai
!python -c "import monai" || pip install -q "monai-weekly[nibabel, tqdm]"
!pip install torch
!pip install tensorboard
!pip install wrench
!pip install smart-open

# load packages
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import os
import nibabel as nib
import nilearn as nil

import boto3
import botocore
from io import BytesIO
import fsspec
import time

from sklearn.metrics import roc_auc_score
import s3fs

import logging
import os
import sys
import shutil
import tempfile

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, ImageDataset, ArrayDataset, Dataset
# need change below to load necessary functionss
from monai.transforms import (
    EnsureChannelFirst,
    EnsureChannelFirstd,
    Compose,
    RandRotate90,
    RandRotate90d,
    Resize,
    Resized,
    ScaleIntensity,
    ScaleIntensityd,
    LoadImage,
    LoadImaged,
    Orientationd,
)
from monaiWrench import NibabelS3Reader

from vit import ViT
from torchWrench import load_checkpoint
import wandb

# gpu
pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
print_config()

##########################
# create a list of s3 dir. details removed.
# get s3 images list
# Create an S3FileSystem object
s3_file = fsspec.filesystem("s3")

# Specify the S3 directory path
s3_directory = "s3://....../ADNI_T1_m06/"

# List all files in the S3 directory
t1 = time.time()
file_paths = s3_file.glob(s3_directory + "**/*.nii*", detail=False)
t2 = time.time()
print(t2-t1)

# what are the files/conditions we would like to process
df_files = pd.read_csv('../....../ADNI_m06_T1.csv')

# select scan type
descs = [#type of scan
]
df_baseline = df_files[df_files.Description.isin(descs)].\
    sort_values(by="Description", key=lambda column: column.map(lambda e: descs.index(e)))
df_baseline = df_baseline.drop_duplicates(subset='Subject', keep='first')
print(df_baseline.shape[0])
print(df_baseline.Subject.nunique(), df_files.Subject.nunique())

## select participants
df_baseline = df_baseline[df_baseline.Group.isin([......])].reset_index(drop=True)

# generate lists of image paths
images = []
haveData= []
labels = []
for idx, row in df_baseline.iterrows():
    subs = row['Image Data ID']
    res = [i for i in file_paths if subs in i]
    if len(res)!=0:
        images.append(res[0].replace("ADNI_T1_Baseline", "ADNI_T1_Baseline_skull_strip")) 
        labels.append('the thing you wanna classify')
        haveData.append(True)
    else:
        haveData.append(False)
labels = np.array(labels) 
labels = torch.nn.functional.one_hot(torch.as_tensor(labels)).float()

ims = [{'img':'s3://'+im} for im in images]
for i in np.arange(len(ims)):
    ims[i]['label'] = labels[i]
##########################
# create train/test
#from monai.transforms import 
####################################
# image transformation/augmentation
####################################

def threshold(x):
    t = np.percentile(x,80)
    return x > t
    
train_transforms = Compose([LoadImaged(keys=['img'], reader=NibabelS3Reader(), image_only=True, ensure_channel_first=True),
                            CropForegroundd(keys=["img"],source_key="img",select_fn=threshold),
####################################
# image transformation/augmentation
####################################
                           ])


val_transforms = Compose([LoadImaged(keys=['img'], reader=NibabelS3Reader(), image_only=True, ensure_channel_first=True),
####################################
# image transformation/augmentation
####################################
                           ])


im_idx = np.random.permutation(len(ims))

# create a training data loader
train_ds = Dataset(data=np.array(ims)[im_idx[:-100]].tolist(), transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=pin_memory)

# create a validation data loader
val_ds = Dataset(data=np.array(ims)[im_idx[-100:]].tolist(), transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=8, num_workers=2, pin_memory=pin_memory)

# set up model
from vit import ViT
# model = ViT(...parameters)
# gpu computation
if torch.cuda.is_available():
    print("GPU is available")
    model.to(device)
    map_loc = None
else:
    print("GPU is not available")
    map_loc = 'cpu'
#load model weight
store_dict = model.state_dict()
# load foundation model weights

filename = '../Data/vit_base-mae-v1.ckpt'

state_dict_before = {
    k: v.clone() for k, v in model.state_dict().items()
}
load_checkpoint(
    model,
    filename,
    map_location=map_loc,
    revise_keys=[(r"^_model\.encoder\._orig_mod\.", "")],
)
state_dict_after = {
    k: v.clone() for k, v in model.state_dict().items()
}
assert any(
    torch.equal(state_dict_before[k], v) == False
    for k, v in state_dict_after.items()
), "State dict has not changed after loading weights"

# get image embeddings from ViT
t1 = time.time()

train_emb = np.empty((0,216,768))
train_emb2 = np.empty((0,216,768))
train_labels = np.empty(0)
val_emb = np.empty((0,216,768))
val_emb2 = np.empty((0,216,768))
val_labels = np.empty(0)

model.eval()
for train_data in train_loader:
    train_images, l = train_data['img'].to(device), train_data['label'].to(device)
    with torch.no_grad():
        train_outputs, hidden_states_out = model(train_images)
    train_emb = np.append(train_emb, hidden_states_out[-1].detach().cpu().numpy(),axis=0)
    train_emb2 = np.append(train_emb2, train_outputs.detach().cpu().numpy(),axis=0)
    train_labels = np.concatenate((train_labels, np.argmax(l.detach().cpu().numpy(), axis=1)))
                          
for val_data in val_loader:
    val_images, l = val_data['img'].to(device), val_data['label'].to(device)
    with torch.no_grad():
        val_outputs, hidden_states_out = model(val_images)
    val_emb = np.append(val_emb, hidden_states_out[-1].detach().cpu().numpy(),axis=0)
    val_emb2 = np.append(val_emb2, val_outputs.detach().cpu().numpy(),axis=0)
    val_labels = np.concatenate((val_labels, np.argmax(l.detach().cpu().numpy(), axis=1)))
t2 = time.time()
print(t2-t1)


#process feature
X = np.concatenate((train_emb,val_emb),axis=0) # pre-norm
# X = np.concatenate((train_emb2,val_emb2),axis=0) # post-norm
X = X.mean(axis=1)
print(X.shape)

y = np.concatenate((train_labels, val_labels))
print(y.shape)


# PCA on subject-wise 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaling=StandardScaler()
X_full = np.concatenate((train_emb,val_emb),axis=0)
X_flatten = X_full.reshape(X_full.shape[0],-1)
print(X_flatten.shape)
# Use fit and transform method 
scaling.fit(X_flatten)
Scaled_data=scaling.transform(X_flatten)
 
# Set the n_components=3
principal=PCA(n_components=100)
principal.fit(Scaled_data)
X_PCA = principal.transform(Scaled_data)

print(X_PCA.shape)

# XGBoost
from sklearn.model_selection import cross_val_score, StratifiedKFold,cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import xgboost as xgb

# Define the XGBoost classifier
model = xgb.XGBClassifier(device="cuda")

# Define the cross-validation strategy
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Perform cross-validation and calculate performance metrics
cv_results = cross_validate(model, X, y, cv=10, scoring=['accuracy', 'roc_auc', 'precision', 'recall', 'f1'])

# Output performance metrics
print("Accuracy:", np.mean(cv_results['test_accuracy']))
print("AUC:", np.mean(cv_results['test_roc_auc']))
print("Precision:", np.mean(cv_results['test_precision']))
print("Recall:", np.mean(cv_results['test_recall']))
print("F1 Score:", np.mean(cv_results['test_f1']))


#logistic
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate,StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

# Generate sample data

# Create the logistic regression model
model = LogisticRegression(penalty='l2', C=10, max_iter=10000)  # L2 regularization

# Perform 10-fold cross-validation
cv_results = cross_validate(model, X, y, cv=10, scoring=['accuracy', 'roc_auc', 'precision', 'recall', 'f1'],return_indices= True)

# Output performance metrics
print("Accuracy:", np.mean(cv_results['test_accuracy']))
print("AUC:", np.mean(cv_results['test_roc_auc']))
print("Precision:", np.mean(cv_results['test_precision']))
print("Recall:", np.mean(cv_results['test_recall']))
print("F1 Score:", np.mean(cv_results['test_f1']))