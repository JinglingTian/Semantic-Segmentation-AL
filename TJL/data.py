import torch
from torch.utils.data import Dataset
from skimage.io import imread
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as tvF
from .utils import *
import os


class MyDataset(Dataset):
    def __init__(self,
                 img_list,
                 mask_list,
                 crop_size=None,
                 erode_iter=0,
                 brightness_factor=None,
                 use_clahe = False
                 ):
        super().__init__()
        self.crop_size = crop_size
        self.erode_iter = erode_iter
        self.img_list = img_list
        self.img_list.sort()
        self.mask_list = mask_list
        self.mask_list.sort()

        # 数据增强
        self.random_brightness = None
        if brightness_factor is not None:
            self.random_brightness = transforms.ColorJitter(brightness_factor[0],brightness_factor[1])

        self.use_clahe = use_clahe

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = min_max_norm(imread(self.img_list[index]))
        # 直方图均衡化
        if self.use_clahe:
            img = hist_clahe(img)
        img = torch.tensor(img).unsqueeze(0).float()


        mask = imread(self.mask_list[index])
        # 腐蚀
        kernel = np.zeros((3,3)).astype(np.uint8)
        kernel[1,:]=1
        kernel[:,1]=1
        mask = cv.erode(mask,kernel,iterations=self.erode_iter)
        
        mask[mask>0]=1
        mask = mask.astype(np.uint8)
        mask = torch.tensor(mask).unsqueeze(0).float()


        # 随机剪裁
        if self.crop_size is not None:
            crop_params = transforms.RandomCrop.get_params(img, self.crop_size)
            img = tvF.crop(img,*crop_params)
            mask = tvF.crop(mask,*crop_params)
        
        # 随机亮度变换
        if self.random_brightness is not None:
            img = self.random_brightness(img)
        


        edge = torch.tensor(get_sobel(mask[0].numpy())).unsqueeze(0).float()
        return img,mask,edge

class ALTrainDataset(Dataset):
    def __init__(self,
                 img_path,
                 mask_path,
                 use_clahe = False
                 ):
        super().__init__()
        self.img_path = img_path
        self.mask_path = mask_path
        self.data_list = []
        for f in os.listdir(mask_path):
            name,h,w = f[:-4].split("_")
            self.data_list.append([name,int(h),int(w)])

        self.use_clahe = use_clahe

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        name,h,w = self.data_list[index]
        img = min_max_norm(imread(os.path.join(self.img_path,f"{name}.tif")))
        mask = min_max_norm(imread(os.path.join(self.mask_path,f"{name}_{h}_{w}.png")))        
        # 直方图均衡化
        if self.use_clahe:
            img = hist_clahe(img)
        img = img[h:h+mask.shape[0],w:w+mask.shape[1]]
        img = torch.tensor(img).unsqueeze(0).float()        

        mask[mask>0]=1
        mask = mask.astype(np.uint8)
        mask = torch.tensor(mask).unsqueeze(0).float()

        edge = torch.tensor(get_sobel(mask[0].numpy())).unsqueeze(0).float()
        return img,mask,edge