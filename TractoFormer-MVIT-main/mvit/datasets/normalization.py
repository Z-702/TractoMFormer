import torch
import torchvision.transforms.functional as TF
from torch.masked import masked_tensor

def normalize(img,mean,std):
    if img.shape[0]!=3:
        raise NotImplementedError
    img1 = ((img-mean)/std)
    img1[img!=0] = 0
    del img
    return img1

class Normalize(object):
    def __init__(self,mean,std):
        if not isinstance(mean,torch.Tensor):
            mean = torch.tensor(mean).reshape(-1,1,1)
        if not isinstance(std,torch.Tensor):
            std = torch.tensor(std).reshape(-1,1,1)
        self.mean = mean
        self.std = std
    def __call__(self,img):
        return normalize(img,self.mean,self.std)
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean},std={self.std})'
    
    