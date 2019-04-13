from torch.utils.data import Dataset
import numpy as np
import torch
import math
import glymur
import torchvision.transforms as transforms
def imread(file, mode='L'):
    assert mode == 'L', 'only load grayscale'
    return glymur.Jp2k(file)[:]
    
from functools import reduce
import os


class FolderDataset(Dataset):

    def __init__(self,
         dir_path,  # e.g. 'samples/'
         transform=None, filter_func = None):

        super(FolderDataset, self).__init__()
        self.dir_path = dir_path
        self.transform = transform
        self.files = sorted(list(map(lambda x: os.path.join(dir_path, x), os.listdir(dir_path))))
        if filter_func != None:
            self.files = list(filter(filter_func, self.files))
        print ('Total files', len(self.files))
    @property
    def shape(self):
        return (len(self),) + self.load_file(0).shape

    def __len__(self):
        return len(self.files)

    def load_file(self, item):
        raise NotImplementedError()

    def __getitem__(self, item):
        datapoint  = self.load_file(item)    
        datapoint = datapoint.astype(np.float32) / 4096.0 * 2 -1       
        
        if self.transform != None:
            datapoint = self.transform(datapoint)  
        
        datapoint = torch.from_numpy(datapoint.astype('float32'))      
        return datapoint


class Jp2ImageFolderDataset(FolderDataset):

    def __init__(self,
                 dir_path='//data/jp2/',
                 imread_mode='L',
         transform=None,filter_func=None,imsize=256):
        self.imread_mode = imread_mode
        self.imsize=imsize
        super(Jp2ImageFolderDataset, self).__init__(dir_path, transform,filter_func)

    def load_file(self, item):
        im = imread(self.files[item], mode=self.imread_mode)
        h, w = im.shape
        factor_h = h // self.imsize
        factor_w = w // self.imsize
        im = im.reshape(1, h // factor_h, factor_h, w // factor_w, factor_w).mean((2, 4), keepdims=False)     
        assert im.ndim == 3
        return im

