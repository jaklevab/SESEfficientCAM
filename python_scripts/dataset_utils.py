# Imports

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
from torch.utils.data import Dataset
from tqdm import tqdm as tqdmn
import torch.nn as nn
import skimage
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
from scipy import misc
import multiprocessing
from joblib import Parallel, delayed

BASE_DIR = "/warehouse/COMPLEXNET/jlevyabi/"
OUTPUT_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/outputs/test/" #esa_URBAN_ATLAS_FR/"
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
CPU_COUNT = multiprocessing.cpu_count()
CPU_FRAC = .6
CPU_USE = int(CPU_FRAC*CPU_COUNT)

def has_file_allowed_extension(filename, extensions):
    """ Check if file has image extension"""
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)    
    
def parallel_folder_extraction(dir,target,extensions,dic_im2target,im2city,null_thresh,city):
    """ 
    Log and filter all files present in subdirectories of dir + target 
    Parameters:
        - dir: Directory with all image subdirectories
        - target: Subdirectory in particular to go through
        - extensions: Permitted file extensions
        - dic_im2target: Dictionary linking image filename to SES label to predict
        - im2city:  Dictionary linking image filename to city boundary where the tile is located (If 'all cities' introduce real value and set city to None)
        - null_thresh: Threshold of dead pixels
        - city: City from which tiles should be considered (If 'all' set to None)
    Returns:
        - List of images to account for in ImageGenerator
    
    """
    images = []
    if city is not None:
        city = city.lower()
    d = os.path.join(dir, target)
    if not os.path.isdir(d):
        return images
    for root, _, fnames in sorted(os.walk(d)):
        for fname in sorted(fnames):
            shorten_fname = fname.split(".")[0].split("_")[-1]
            if has_file_allowed_extension(fname, extensions) and shorten_fname in dic_im2target:
                if city is not None:
                    if (not(fname.lower() in im2city) or not(im2city[fname.lower()].lower() == city)): 
                        continue
                elif im2city is not None:
                    if not(fname.lower() in im2city):
                        continue
                path = os.path.join(root, fname)
                target = dic_im2target[shorten_fname]
                image = misc.imread(path)
                if np.isnan(target) or 100*(image==0).sum()/image.size > null_thresh :
                    continue
                item = (path, target)
                images.append(item)
    return images

def parallel_make_dataset(dir, dic_im2target, im2city, extensions, city, null_thresh = 10):
    dirs_to_treat = sorted(os.listdir(dir))
    nb_dirs = len(dirs_to_treat)
    pre_full = Parallel(n_jobs=min(CPU_USE,nb_dirs))(delayed(parallel_folder_extraction)(dir=dir,target=target,extensions=extensions,dic_im2target=dic_im2target,im2city=im2city,null_thresh=null_thresh,city=city)
                for target in tqdmn(dirs_to_treat))
    return [data for pre in pre_full for data in pre]
     
    
def make_dataset(dir, dic_im2target, extensions, city, null_thresh = 10):
    print("Needs to be changed to fit parallel_folder_extraction")
    sys.exit()
    images = []
    naned_targets, total = 0, 0
    dir = os.path.expanduser(dir)
    print("Loading SubDirs dataset")
    for target in tqdmn(sorted(os.listdir(dir))):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    target = dic_im2target[fname.split(".")[0].split("_")[-1]]
                    image = misc.imread(path)
                    total+=1
                    if np.isnan(target) or 100*(image==0).sum()/image.size > null_thresh :
                        naned_targets+=1
                        continue
                    item = (path, target)
                    images.append(item)
    print("Omitted %d entries containing null/nan source/targets out of %d entries"%(naned_targets,total))
    return images

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class DigitalGlobeFrance(Dataset):
    def __init__(self, root, im2target, extensions=IMG_EXTENSIONS,
                 loader=default_loader, transform=None, target_transform=None, im2city=None, city=None):
        
        if not(city is None):
            assert im2city is not None,"City is not none please enter im2city dictionary"

        images = parallel_make_dataset(root, dic_im2target=im2target, im2city=im2city, extensions=extensions, city=city)        
        
        if len(images) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        self.root = root
        self.loader = loader
        self.extensions = extensions
        self.images = images
        self.im2target = im2target
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, target = self.images[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def train_test_split(self, frac = .2, shuffle_dataset = True, seed = 0):
        dataset_size = len(self)
        indices = list(range(dataset_size))
        split_val = int(np.floor(.5 * frac * dataset_size))
        split_test = int(np.floor(frac * dataset_size))
        if shuffle_dataset :
            np.random.seed(seed)
            np.random.shuffle(indices)
        train_indices, val_indices, test_indices = indices[split_test:], indices[:split_val], indices[split_val:split_test]
        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        return train_sampler, valid_sampler, test_sampler
