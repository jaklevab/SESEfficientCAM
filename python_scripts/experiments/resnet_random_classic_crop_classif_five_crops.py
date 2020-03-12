# Library Imports
import torch
import torch.nn as nn
import torchvision
import numpy as np
import torchvision.models as models
from torchvision import transforms, utils,datasets
import torch.utils.data as data
from PIL import Image
import os
import os.path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm as tqdmn
import torch.nn as nn
from sklearn.preprocessing import (MinMaxScaler,StandardScaler,)
import skimage
import matplotlib.pyplot as plt
from collections import OrderedDict
from decimal import Decimal
import sys
import pandas as pd
import sklearn.metrics as metrics
from tensorboardX import SummaryWriter


NB_CLASSES = 5
base_dir = "/warehouse/COMPLEXNET/jlevyabi/"
sys.path.append(base_dir + "SATELSES/equirect_proj_test/cnes/python_files/")
from dataset_utils import DigitalGlobeFrance
from model_utils import train, test , load_checkpoint, save_checkpoint

# Data Imports
sat_dir = base_dir + "SATELSES/equirect_proj_test/cnes/data_files/esa/URBAN_ATLAS/"
census_dir = base_dir + 'REPLICATE_LINGSES/data_files/census_data/'
ua_dir = base_dir + "SATELSES/equirect_proj_test/cnes/data_files/land_ua_esa/FR/"
output_dir = base_dir + "SATELSES/equirect_proj_test/cnes/data_files/outputs/esa_URBAN_ATLAS_FR/"
log_dir = output_dir + "../../python_files/log_files/"
writer = SummaryWriter(log_dir=log_dir + "resnet50_five_crops")

print("Prepping dic_im2target")
pre_dic = pd.read_csv(output_dir + "../census_data/squares_to_ses.csv" )
pre_dic.dropna(axis=0,subset=["income"],inplace=True)
#income = pre_dic.income
income = pre_dic.income
class_thresholds = [np.percentile(income,k) for k in np.linspace(0,100,NB_CLASSES +1 )]
x_to_class = np.digitize(income,class_thresholds)
x_to_class[x_to_class==np.max(x_to_class)] = NB_CLASSES
pre_dic["treated_income"] = x_to_class - 1
dic_im2target = {idINSPIRE:inc_vals for idINSPIRE,inc_vals in pre_dic[["idINSPIRE","treated_income"]].values}

# Data Preparation3
print("Loading Data")
w_desired = h_desired = 224
nb_channels = 3
batch_sz = 16
transform = transforms.Compose(
    [transforms.ColorJitter(),
     transforms.Compose([
         transforms.FiveCrop((w_desired,h_desired)),
         transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
     ])])

dg_dataset = DigitalGlobeFrance(output_dir, dic_im2target, transform=transform)
train_idx, val_idx, test_idx = dg_dataset.train_test_split(frac=0.4)
train_loader = DataLoader(dg_dataset, batch_size=batch_sz, num_workers=12, sampler=train_idx)
val_loader = DataLoader(dg_dataset, batch_size=batch_sz, num_workers=12, sampler=val_idx)
test_loader = DataLoader(dg_dataset, batch_size=batch_sz, num_workers=12, sampler=test_idx)


print("Done Loading")
# Model Parameters
lr = 1e-3
#momentum = 0.9
#weight_decay = 1.0e-3
start_epoch, max_epochs = (0, 10)
gamma = 0.1
st_size = 2
ckpt_dir = output_dir + "../model_data/resnet_random_classic_crop/"

# Model Definition
print("Defining Model")
model = models.resnet50(pretrained=True)
last_layer_in_fts = list(model.children())[-1].in_features
model.fc = nn.Linear(last_layer_in_fts,out_features=NB_CLASSES,bias=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
optimizer = torch.optim.Adam(model.parameters(), lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=gamma, step_size=st_size)
metrics = metrics.accuracy_score

# Model Training
print("Training Model")
train_test_model, train_test_state = train(model,train_loader,val_loader,
                                           criterion,optimizer,metrics, scheduler,
                                           device,ckpt_dir,num_epochs=max_epochs,writer=writer,multiple_crops=True)
writer.close()
# Model Testing
print("Testing Model")
test_loader = DataLoader(dg_dataset, batch_size=batch_sz, num_workers=4, sampler=test_idx)
test_score = test(test_loader, model, criterion, metrics, device)
