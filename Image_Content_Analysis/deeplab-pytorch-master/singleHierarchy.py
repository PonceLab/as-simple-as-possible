#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   07 January 2019

from __future__ import absolute_import, division, print_function

import click
import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from libs.models import *
from libs.utils import DenseCRF
import numbers
import os
import re
import shutil



def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def get_classtable(CONFIG):
    with open(CONFIG.DATASET.LABELS) as f:
        classes = {}
        for label in f:
            label = label.rstrip().split("\t")
            classes[int(label[0])] = label[1].split(",")[0]
    return classes


def setup_postprocessor(CONFIG):
    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=CONFIG.CRF.ITER_MAX,
        pos_xy_std=CONFIG.CRF.POS_XY_STD,
        pos_w=CONFIG.CRF.POS_W,
        bi_xy_std=CONFIG.CRF.BI_XY_STD,
        bi_rgb_std=CONFIG.CRF.BI_RGB_STD,
        bi_w=CONFIG.CRF.BI_W,
    )
    return postprocessor


def preprocessing(image, device, CONFIG):
    # Resize
    scale = CONFIG.IMAGE.SIZE.TEST / max(image.shape[:2])
    image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
    raw_image = image.astype(np.uint8)

    # Subtract mean values
    image = image.astype(np.float32)
    image -= np.array(
        [
            float(CONFIG.IMAGE.MEAN.B),
            float(CONFIG.IMAGE.MEAN.G),
            float(CONFIG.IMAGE.MEAN.R),
        ]
    )

    # Convert to torch.Tensor and add "batch" axis
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    image = image.to(device)

    return image, raw_image


def inference(model, image, raw_image=None, postprocessor=None):
    _, _, H, W = image.shape

    # Image -> Probability map
    logits = model(image)
    logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    probs = F.softmax(logits, dim=1)[0]
    probs = probs.cpu().numpy()

    # Refine the prob map with CRF
    if postprocessor and raw_image is not None:
        probs = postprocessor(raw_image, probs)

    labelmap = np.argmax(probs, axis=0)

    return labelmap


def inferenceHierarchy(model, image, raw_image=None, postprocessor=None, sizeThresh=1/9, nIterations=10):
    _, _, H, W = image.shape

    # Image -> Probability map
    logits = model(image)
    logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    probs = F.softmax(logits, dim=1)[0]
    probs = probs.cpu().numpy()
    probsList = []
    probsList.append(probs)

    labelmapList=[]
    
    for ndx in range(0,int(nIterations)):
        # Refine the prob map with CRF
        if postprocessor and raw_image is not None:
            probs2 = postprocessor(raw_image, probs)
            
        labelmap2 = np.argmax(probs2, axis=0)
        labelmapList.append(labelmap2)
        labels = np.unique(labelmap2)
        hasBigSegs=False
        if ndx!=0:
            probsList.append(probs)
        for label in labels:
            if (np.sum(labelmap2==label)/labelmap2.size)>sizeThresh: # and label!=0:
                probs[label,:,:]=probs.min()
                hasBigSegs=True
        probs=probs/np.sum(probs,(0))
        
        if not hasBigSegs:
            break
    
    return labelmapList,probsList

def singleHierarchy(config_path, model_path, image_path, cuda, crf, sizeThresh=1/9, nIterations=10, doPlot=True):
    """
    Inference from a single image
    """

    # Setup
    CONFIG = OmegaConf.load(config_path)
    device = get_device(cuda)
    torch.set_grad_enabled(False)

    classes = get_classtable(CONFIG)
    postprocessor = setup_postprocessor(CONFIG) if crf else None

    model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print("Model:", CONFIG.MODEL.NAME)

    # Inference
    if isinstance(image_path,str):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            (path2Im,imName)=os.path.split(image_path)
            (imName2,imExt)=os.path.splitext(imName)
            imNameSimple=re.sub(r'[^A-Za-z0-9]+', '', imName2)+str(np.random.randint(10**6))+imExt
            shutil.copy(image_path,imNameSimple)
            image = cv2.imread(imNameSimple, cv2.IMREAD_COLOR)
            os.remove(imNameSimple)
    else:
        image=np.array(image_path)      
    image, raw_image = preprocessing(image, device, CONFIG)
    # labelmap = inference(model, image, raw_image, postprocessor)
    labelmapList,probsList = inferenceHierarchy(model, image, raw_image, postprocessor, sizeThresh, nIterations)
    if doPlot:
        for labelmap in labelmapList:
            labels = np.unique(labelmap)
        
            # Show result for each class
            rows = np.floor(np.sqrt(len(labels) + 1))
            cols = np.ceil((len(labels) + 1) / rows)
        
            plt.figure(figsize=(10, 10))
            ax = plt.subplot(rows, cols, 1)
            ax.set_title("Input image")
            ax.imshow(raw_image[:, :, ::-1])
            ax.axis("off")
        
            for i, label in enumerate(labels):
                mask = labelmap == label
                ax = plt.subplot(rows, cols, i + 2)
                ax.set_title(classes[label])
                ax.imshow(raw_image[..., ::-1])
                ax.imshow(mask.astype(np.float32), alpha=0.5)
                ax.axis("off")
        
            plt.tight_layout()
            plt.show()
    else:
        return labelmapList,probsList,classes



#single(r".\configs\cocostuff164k.yaml",r"C:\Users\ponce\Desktop\CarlosSetUpFilesHere\CompressionPaperReviewResponse\resources\deeplab-pytorch-master\data\models\coco\deeplabv1_resnet101\caffemodel\deeplabv2_resnet101_msc-cocostuff164k-100000.pth",r"image.jpg",True,True)

#python demo.py single --config-path .\configs\voc12.yaml --model-path "C:\Users\ponce\Desktop\CarlosSetUpFilesHere\CompressionPaperReviewResponse\resources\deeplab-pytorch-master\data\models\voc12\deeplabv2_resnet101_msc\caffemodel\deeplabv2_resnet101_msc-vocaug.pth" --image-path image.jpg
#python demo.py single --config-path .\configs\cocostuff164k.yaml --model-path "C:\Users\ponce\Desktop\CarlosSetUpFilesHere\CompressionPaperReviewResponse\resources\deeplab-pytorch-master\data\models\coco\deeplabv1_resnet101\caffemodel\deeplabv2_resnet101_msc-cocostuff164k-100000.pth" --image-path image.jpg

if __name__ == "__main__":
    singleHierarchy()
