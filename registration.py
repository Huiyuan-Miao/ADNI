# for demonstration purpose, interaction with amazon S3 and conducting registration in MRI, contents related to actual analysis are removed

# packages needed
!pip install deepbet
!pip install pyqt5
!pip install timm
!pip install einops
!pip install nibabel
!pip install nilearn
!pip install xgboost
!pip install fsspec==2023.6.0
!pip install s3fs==2023.6.0
!pip install boto3==1.34.51
!pip install botocore==1.34.51

# load package
from deepbet import run_bet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
import nilearn as nil
import botocore
from io import BytesIO
import fsspec
import time
import s3fs

import logging
import shutil
import subprocess  # nosec

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from monai.transforms import LoadImage
from monai.visualize.utils import matshow3d
import os.path

#set up
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3_config = Config(retries={"max_attempts": 15, "mode": "standard"})
s3 = boto3.client("s3", config=s3_config)

##########################
# create a list of s3 dir. details removed. variable: images (a list of ADNI MRI scans)
# get s3 images list to be processed
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

# some matching between what I want and what is on S3, select participants that have MRI scans available

# generate lists of image paths
images = []
haveData= []
for idx, row in df_baseline.iterrows():
    subs = row['Image Data ID']
    res = [i for i in file_paths if subs in i]
    if len(res)!=0:
        images.append(res[0].replace("ADNI_T1_Baseline", "ADNI_T1_Baseline_skull_strip")) 
        haveData.append(True)
    else:
        haveData.append(False)
##########################

# download file from s3,registration, upload file to s3
# image loader
l = LoadImage()

# make local dir on AWS sagemaker local environment to hold data downloaded from S3
local_dir = './original_data_skull_strip'
output_dir = './output_data_skull_strip_registerMNI1mm'
template_dir = '../Data/MNI152_T1_1mm_brain.nii.gz'
os.makedirs(local_dir,exist_ok=True)
os.makedirs(output_dir,exist_ok=True)
for i in range(0,len(images)):
    im_dir = output_dir+'/'+file_name[i]
    # just to make sure I don't process the same file twice
    if os.path.isfile(im_dir.replace('.nii','_1.png')) is False:
        print(i)
        # download file from s3
        s3_uri = 's3://'+images[i]
        bucket, prefix = s3_uri[5:].split("/", 1)
        
        try:
            paginator = s3.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        except ClientError as e:
            logger.error(f"Error listing objects in {s3_uri}: {e}")
            raise
    
        downloaded_files = []
    
        for page in pages:
            for obj in page["Contents"]:
                if obj["Key"].endswith(".nii"):
                    local_file_path = os.path.join(local_dir, os.path.basename(obj["Key"]))
                    try:
                        s3.download_file(bucket, obj["Key"], local_file_path)
                    except ClientError as e:
                        logger.error(f"Error downloading {obj['Key']} from {s3_uri}: {e}")
                        raise
                    else:
                        pass
                        # logger.info(f"Downloaded {obj['Key']} to {local_file_path}")
                    downloaded_files.append(local_file_path)
                    output_dir_path = os.path.join(output_dir, os.path.basename(obj["Key"]))
    
        # registration
        t1 = time.time()
        input_paths = [local_dir+'/'+file_name[i]]
        brain_paths = [output_dir+'/'+file_name[i]]
        moving_img = nib.load(input_paths[0])
        template_img = nib.load(template_dir)
        moving_data = moving_img.get_fdata()
        moving_affine = moving_img.affine
        template_data = template_img.get_fdata()
        template_affine = template_img.affine
        identity = np.eye(4)
        affine_map = AffineMap(identity,template_data.shape, template_affine, moving_data.shape, moving_affine)
        resampled = affine_map.transform(moving_data)
        # regtools.overlay_slices(template_data, resampled, None, 0,"Template", "Moving")
        # regtools.overlay_slices(template_data, resampled, None, 1,"Template", "Moving")
        # regtools.overlay_slices(template_data, resampled, None, 2,"Template", "Moving")
        
        # The mismatch metric
        nbins = 32
        sampling_prop = None
        metric = MutualInformationMetric(nbins, sampling_prop)
        # The optimization strategy
        level_iters = [10, 10, 5]
        sigmas = [3.0, 1.0, 0.0]
        factors = [4, 2, 1]
        affreg = AffineRegistration(metric=metric,
                                     level_iters=level_iters,
                                     sigmas=sigmas,
                                     factors=factors)
        transform = TranslationTransform3D()
        params0 = None
        translation = affreg.optimize(template_data, moving_data, transform, params0,
                                       template_affine, moving_affine)
        
        transformed = translation.transform(moving_data)
        # regtools.overlay_slices(template_data, transformed, None, 0,"Template", "Transformed")
        # regtools.overlay_slices(template_data, transformed, None, 1,"Template", "Transformed")
        # regtools.overlay_slices(template_data, transformed, None, 2,"Template", "Transformed")
        
        transform = RigidTransform3D()
        rigid = affreg.optimize(template_data, moving_data, transform, params0,
                                 template_affine, moving_affine,
                                 starting_affine=translation.affine)
        transformed = rigid.transform(moving_data)
        transform = AffineTransform3D()
        # Bump up the iterations to get an more exact fit
        affreg.level_iters = [1000, 1000, 100]
        affine = affreg.optimize(template_data, moving_data, transform, params0,
                                  template_affine, moving_affine,
                                  starting_affine=rigid.affine)
        transformed = affine.transform(moving_data)
        # regtools.overlay_slices(template_data, transformed, None, 0,"Template", "Transformed")
        # regtools.overlay_slices(template_data, transformed, None, 1,"Template", "Transformed")
        # regtools.overlay_slices(template_data, transformed, None, 2,"Template", "Transformed")
        mappedImg = nib.Nifti1Image(transformed, moving_img.affine, moving_img.header)
        nib.save(mappedImg,brain_paths[0])
        t2 = time.time()
        print(t2-t1)
        # visualize results for quality check   
        img = l(brain_paths[0])
        matshow3d(img, fig=None, title=None, figsize=(5,5),
                frames_per_row=None, frame_dim=-1, channel_dim=None, vmin=None,
                vmax=None, every_n=5, interpolation='none', show=False, margin=1)
        plt.savefig(brain_paths[0].replace('.nii','_1'))
        matshow3d(img, fig=None, title=None, figsize=(5,5),
                frames_per_row=None, frame_dim=-2, channel_dim=None, vmin=None,
                vmax=None, every_n=5, interpolation='none', show=False, margin=1)
        plt.savefig(brain_paths[0].replace('.nii','_2'))
        matshow3d(img, fig=None, title=None, figsize=(5,5),
                frames_per_row=None, frame_dim=-3, channel_dim=None, vmin=None,
                vmax=None, every_n=5, interpolation='none', show=False, margin=1)
        plt.savefig(brain_paths[0].replace('.nii','_3'))
        
        regtools.overlay_slices(template_data, transformed, None, 0,"Template", "Transformed")
        plt.savefig(brain_paths[0].replace('.nii','_1_overlay'))
        
        regtools.overlay_slices(template_data, transformed, None, 1,"Template", "Transformed")
        plt.savefig(brain_paths[0].replace('.nii','_2_overlay'))
        
        regtools.overlay_slices(template_data, transformed, None, 2,"Template", "Transformed")
        plt.savefig(brain_paths[0].replace('.nii','_3_overlay'))
                                                                                                                                                                                    
        # upload file to the corresponding folder
        s3 = boto3.client("s3", config=s3_config)
        nifti_uri = s3_uri.replace("ADNI_T1_Baseline_skull_strip", "ADNI_T1_Baseline_skull_strip_registeredMNI1mm")
        bucket, prefix = nifti_uri.replace("s3://", "").split("/", 1)
        status = []
    
        # key = os.path.join(prefix, os.path.basename(output_dir_path))
        key = prefix
        try:
            s3.upload_file(output_dir_path, bucket, key)
        except ClientError as e:
            logger.error(f"Error uploading {output_dir_path} to {nifti_uri}: {e}")
            status.append(False)
        else:
            logger.info(f"Uploaded {output_dir_path} to {nifti_uri}")
            status.append(True)
        # delete file in sagemaker
        os.remove(output_dir_path)
        os.remove(local_file_path)