import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

__EPS__ = 1e-9

# mIoU
def get_miou(pred,mask):
    pred = pred.view(-1)
    mask = mask.view(-1)
    TP = ((pred==1)&(mask==1)).sum()
    FP = ((pred==1)&(mask==0)).sum()
    TN = ((pred==0)&(mask==0)).sum()
    FN = ((pred==0)&(mask==1)).sum()
    miou = 0.5*(TP/(FN+FP+TP+__EPS__) + TN/(FN+FP+TN+__EPS__))
    return miou

# Dice
def get_dice(pred,mask):
    pred = pred.view(-1)
    mask = mask.view(-1)
    TP = ((pred==1)&(mask==1)).sum()
    FP = ((pred==1)&(mask==0)).sum()
    FN = ((pred==0)&(mask==1)).sum()
    
    dice = (2*TP)/(2*TP+FP+FN+__EPS__)
    return dice

# Precision
def get_pre(pred,mask):
    pred = pred.view(-1)
    mask = mask.view(-1)
    TP = ((pred==1)&(mask==1)).sum()
    FP = ((pred==1)&(mask==0)).sum()
    
    pre = TP/(TP+FP+__EPS__)
    return pre

# Accuracy
def get_acc(pred,mask):
    pred = pred.view(-1)
    mask = mask.view(-1)
    TP = ((pred==1)&(mask==1)).sum()
    FP = ((pred==1)&(mask==0)).sum()
    TN = ((pred==0)&(mask==0)).sum()
    FN = ((pred==0)&(mask==1)).sum()
    
    acc = (TP+TN)/(TP+FP+TN+FN+__EPS__)
    return acc

# Recall
def get_recall(pred,mask):
    pred = pred.view(-1)
    mask = mask.view(-1)
    TP = ((pred==1)&(mask==1)).sum()
    FN = ((pred==0)&(mask==1)).sum()
    
    recall = (TP)/(TP+FN+__EPS__)
    return recall
    
# min_max归一化
def min_max_norm(x):
    if x.max()-x.min()==0:
        return x*0
    return (x-x.min())/(x.max()-x.min())

# 计算模型参数量
def model_param_count(model):
    return sum([p.numel() for p in model.parameters()])



# Sobel提取边缘
def get_sobel(img):
    sobel_x = cv.convertScaleAbs(cv.Sobel(img,-1,1,0))
    sobel_y = cv.convertScaleAbs(cv.Sobel(img,-1,0,1))
    sobel_xy = cv.addWeighted(sobel_x,0.5,sobel_y,0.5,0)
    sobel_xy[sobel_xy>0]=1
    return sobel_xy


# 直方图均衡化CLAHE 输入归一化图像numpyarray
def hist_clahe(img,clipLimit=1.5, tileGridSize=(4,4)):
    clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img = clahe.apply((img*255).astype(np.uint8))
    img = min_max_norm(img.astype(np.float32))
    return img



# 显示图片
def imshow(img_list,xyhw=None,save_path=None,show=True):
    img_numbers = len(img_list)
    plt.figure(figsize=(6*img_numbers,6*2))
    for i,img in enumerate(img_list):
        plt.subplot(2,img_numbers,i+1)
        plt.imshow(img,cmap="gray")
    if xyhw is not None:
        x,y,h,w = xyhw
        for i,img in enumerate(img_list):
            plt.subplot(2,img_numbers,img_numbers+i+1)
            plt.axis("off")
            plt.imshow(img[x:x+h,y:y+w],cmap="gray")
    
    if save_path is not None:
        plt.savefig(save_path,bbox_inches='tight')
    
    if not show:
        plt.close()
    else:
        plt.show()

# 展示分割结果    img: 原图像(3,512,512); mask: 原mask图像(512,512); pred_mask: 分割结果(512,512) 
def result_show(img,mask,pred_mask):
    plt.figure(figsize=(10,3))
    plt.subplot(1,3,1)
    plt.title("img")
    plt.imshow(img,cmap="gray")
    plt.subplot(1,3,2)
    plt.title("mask")
    plt.imshow(mask,cmap="RdYlBu_r")
    plt.subplot(1,3,3)
    plt.title("pred")
    plt.imshow(pred_mask,cmap="RdYlBu_r")
    plt.show()
    