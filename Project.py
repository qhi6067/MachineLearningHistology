import sklearn as sk 
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn 
import torchvision 
from torchvision import datasets, models, transforms
import time 
import matplotlib.pyplot as plt 
import os 
from PIL import Image 
from tempfile import TemporaryDirectory

cudnn.benchmark = True 
plt.ion()#interactive mode on, comment out to remove

#Loading the data: 

#Normalizing images: 
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229, 0.224, 0.225])
    ]),
    'val':transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229, 0.224, 0.225])
    ]),
}

#directory of extracted data (entire folder):
data_dir = 'EmptyForNow'
image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train', 'val']}
datasetsSize = {x:len(image_datasets[x]) for x in ['train','val']}
classNames = image_datasets['train'].classes 

device = torch.device('cuda:0' if torch.cuda.is_available() else"cpu")