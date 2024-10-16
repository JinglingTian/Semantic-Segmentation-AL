import torch
import torch.nn as nn
import cv2 as cv
import numpy as np
from TJL.utils import min_max_norm

class Edge_BCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def get_edge_weight(self,edge):
        _,_,H,W = edge.shape
        edge_list = edge.squeeze(1).long().cpu().numpy()
        weights = []
        for e in edge_list:
            dist = cv.distanceTransform(e.astype(np.uint8),cv.DIST_L2,maskSize=0)
            dist = min_max_norm(dist)
            weights.append(torch.from_numpy(dist).view(1,1,H,W))
        weights = torch.concat(weights,0).to(edge.device)
        return weights

    def forward(self,pred,pred_edge,label,edge):
        W = self.get_edge_weight(edge)
        label_loss = (-(label*torch.log(pred)+(1-label)*torch.log(1-pred)) * (1+W)).mean()
        edge_loss = (-(edge*torch.log(pred_edge)+(1-edge)*torch.log(1-pred_edge))).mean()
        return label_loss+edge_loss


class BCELoss(nn.Module):
    def __init__(self,alpha=1,beta=1) -> None:
        super().__init__()
        self.alpha=alpha
        self.beta=beta

    def forward(self,pred,pred_edge,label,edge):
        label_loss = (-(label*torch.log(pred)+(1-label)*torch.log(1-pred))).mean()
        edge_loss = (-(edge*torch.log(pred_edge)+(1-edge)*torch.log(1-pred_edge))).mean()
        return self.alpha*label_loss + self.beta*edge_loss


