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
from tensorboardX import SummaryWriter
from sklearn import metrics
base_dir = "/warehouse/COMPLEXNET/jlevyabi/"
sys.path.append(base_dir + "SATELSES/equirect_proj_test/cnes/python_files/")
import pandas as pd
from dataset_utils import DigitalGlobeFrance
from model_utils import train, test , load_checkpoint, save_checkpoint

# Data Imports
sat_dir = base_dir + "SATELSES/equirect_proj_test/cnes/data_files/esa/URBAN_ATLAS/"
census_dir = base_dir + 'REPLICATE_LINGSES/data_files/census_data/'
ua_dir = base_dir + "SATELSES/equirect_proj_test/cnes/data_files/land_ua_esa/FR/"
output_dir = base_dir + "SATELSES/equirect_proj_test/cnes/data_files/outputs/esa_URBAN_ATLAS_FR/"
python_dir = base_dir + "SATELSES/equirect_proj_test/cnes/python_files/experiments/"
log_dir = output_dir + "../../python_files/log_files/"

NB_CLASSES = 5

print("Prepping dic_im2target")
pre_dic = pd.read_csv(output_dir + "../census_data/squares_to_ses.csv" )
pre_dic.dropna(axis=0,subset=["income"],inplace=True)
income = pre_dic.income
class_thresholds = [np.percentile(income,k) for k in np.linspace(0,100,NB_CLASSES +1 )]
x_to_class = np.digitize(income,class_thresholds)
x_to_class[x_to_class==np.max(x_to_class)] = NB_CLASSES
pre_dic["treated_income"] = x_to_class - 1
dic_im2target = {idINSPIRE:inc_vals for idINSPIRE,inc_vals in pre_dic[["idINSPIRE","income"]].values}

print("im2city")
pre_city_dic = pd.read_csv(ua_dir + "../../sources/census_cells_city_boundary.csv" )
pre_city_dic.dropna(axis=0,subset=["city"],inplace=True)
im2city = {("FR_URBANATLAS_200m_" + idINSPIRE + ".png").lower():city for idINSPIRE,city in pre_city_dic[["idINSPIRE","city"]].values}

classic_resnet_size = 224
batch_sz = 24
nb_channels = 3
cpu_counts = os.cpu_count()
cpu_dataloader = int(0.5*cpu_counts)

params = pd.read_csv(python_dir + "params.csv",sep=",")
# [(desired_size, crop, multiple_crops, pretrained )]

for _,row in params.iterrows():
    w_desired = h_desired = row.desired_size
    transform_lists = []

    # Color Channel Mods
    if row.jitter:
        transform_lists.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))

    # Size Mod
    if row.crop:
        if row.multiple_crops:
            transform_lists.append(transforms.FiveCrop((w_desired,h_desired)))
        else:
            transform_lists.append(transforms.RandomCrop((w_desired,h_desired)))
    else:
        transform_lists.append(transforms.Resize((w_desired,h_desired)))

    # Tensorize
    transform_lists.append(transforms.ToTensor())
    transform = transforms.Compose(transform_lists)

    # Generate Dataset
    dg_dataset = DigitalGlobeFrance(output_dir, dic_im2target, im2city=im2city, transform=transform)
    train_idx, val_idx, test_idx = dg_dataset.train_test_split(frac=0.4)
    train_loader = DataLoader(dg_dataset, batch_size=batch_sz, num_workers=cpu_dataloader, sampler=train_idx)
    val_loader = DataLoader(dg_dataset, batch_size=batch_sz, num_workers=cpu_dataloader, sampler=val_idx)
    test_loader = DataLoader(dg_dataset, batch_size=batch_sz, num_workers=cpu_dataloader, sampler=test_idx)
    print("Done Loading")

    # Model Parameters
    lr = 1e-3
    momentum = 0.9
    weight_decay = 1.0e-3
    start_epoch, max_epochs = (0, 20)
    gamma = 0.1
    st_size = 5
    ckpt_dir = output_dir + "../model_data/resnet_random_classic_crop/"

    # Model Definition
    print("Defining Model")
    model = models.resnet50(pretrained=row.pretrained)
    last_layer_in_fts = list(model.children())[-1].in_features
    #model.fc = nn.Linear(last_layer_in_fts,out_features=NB_CLASSES,bias=True)
    model.fc = nn.Linear(last_layer_in_fts,out_features=1,bias=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=gamma, step_size=st_size)
    #metrics = metrics.accuracy_score
    mymetrics = metrics.r2_score
    writer = SummaryWriter(log_dir=log_dir + "resnet50_cities_regression_jitter_%d_desired_size_%d_crop_%d_multiple_%d_pretrained_%d"%(int(row.jitter),int(row.desired_size),int(row.crop),
                                                                                                                         int(row.multiple_crops),int(row.pretrained)))

    # Model Training
    print("Training Model")
    train_test_model, train_test_state = train(model,train_loader,val_loader,
                                               criterion,optimizer,mymetrics, scheduler,
                                               device,ckpt_dir,num_epochs=max_epochs,writer=writer,multiple_crops=row.multiple_crops)
    writer.close()

    # Model Testing
    print("Testing Model")
    test_score = test(test_loader, model, criterion, mymetrics, device, multiple_crops=row.multiple_crops)
