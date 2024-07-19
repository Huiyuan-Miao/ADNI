# for demonstration purpose, interaction with amazon S3, conducting data analysis, ViT training, contents related to actual analysis are removed, generic code that can be adapted to any conditions
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

# set up model training details...
lr = 1e-4
n_epochs = 50
loss_function = torch.nn.BCEWithLogitsLoss()  # also works with this data
# loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr) # no LR decay
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader),epochs=n_epochs, div_factor=25)

# start a typical PyTorch training
val_interval = 1
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
auc_values = []
writer = SummaryWriter()
max_epochs = n_epochs
val_losses = []
t1 = time.time()
for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data['img'].to(device), batch_data['label'].to(device)
        # print(labels)
        optimizer.zero_grad()
        outputs,_ = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        # print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()

        vals = []
        vll = []
        val_loss = 0
        num_correct = 0.0
        metric_count = 0
        step = 0
        for val_data in val_loader:
            step += 1
            val_images, val_labels = val_data['img'].to(device), val_data['label'].to(device)
            with torch.no_grad():
                val_outputs,_ = model(val_images)
                value = torch.eq(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))
                # value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                metric_count += len(value)
                num_correct += value.sum().item()
            vals = np.concatenate((vals, val_outputs.cpu().detach().numpy()[:,1]))
            vll = np.concatenate((vll, val_labels.argmax(dim=1).cpu().detach().numpy()))
            
            loss = loss_function(val_outputs, val_labels)
            val_loss += loss.item()
        val_losses.append(val_loss/step)
        print(f"average val loss: {val_loss/step:.4f}")
            

        metric = num_correct / metric_count
        metric_values.append(metric)
        auc_val = roc_auc_score(vll, vals)
        auc_values.append(auc_val)
        

        if metric > best_metric:
            best_metric = metric
            best_auc = auc_val
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
            print("saved new best metric model")

        print(f"Current epoch: {epoch+1} current accuracy: {metric:.4f} ")
        print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
        writer.add_scalar("val_accuracy", metric, epoch + 1)
        
        print(f"Current epoch: {epoch+1} current ROC: {auc_val:.4f} ")
        print(f"Best ROC: {best_auc:.4f} at epoch {best_metric_epoch}")
        writer.add_scalar("val_auc", auc_val, epoch + 1)

print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
writer.close()
t2 = time.time()
print(t2-t1)

# plot loss
plt.plot(epoch_loss_values)
plt.plot(np.arange(0,len(epoch_loss_values),1), val_losses)
plt.legend(['train loss','val loss'])
plt.show()