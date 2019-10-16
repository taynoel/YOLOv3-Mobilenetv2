import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import random
import imgUtils
import numpy as np

class ListDataset(Dataset):
    def __init__(self,opt):
        self._opt=opt
        
        with open(self._opt.trainDatasetListFile,"r") as fil:
            filSt=fil.read().replace(";","").split("\n")    
        self.datasetDir=self._opt.trainDatasetDirectory
        self.datasetLabelDir=self._opt.trainDatasetLabelDirectory
        self.imgDataList=filSt
        self.labelDataList=[f.replace(".jpg",".txt") for f in filSt]
        self._len=len(filSt)
        self.randomize=True
        
        
    def __getitem__(self, index):
        if self.randomize:
            #flip or not?
            _flipFlag=random.choice([0,1])
            #randomize padding
            _hr,_wr=random.random(),random.random()
        else:
            _flipFlag,_hr,_wr = 0, 0.5, 0.5
        img = transforms.ToTensor()(Image.open(self.datasetDir+self.imgDataList[index]).convert('RGB'))
        
        img,pad,[ori_h, ori_w] = imgUtils.imgTransformSingleImg(img,_flipFlag,_hr,_wr,self._opt.imgSquareSize)
        _, padded_h, padded_w = img.shape
        
        DataFlag=1
        try:
            _dat=np.loadtxt(self.datasetLabelDir+self.labelDataList[index])
            if _dat.shape[0]==0: DataFlag=0
        except:
            print("Cant read label file ",self.labelDataList[index])
            DataFlag=0
        
        if DataFlag:
            boxes = torch.from_numpy(_dat.reshape(-1, 5))
            if _flipFlag:
                boxes[:, 1]=1-boxes[:, 1]
            boxes[:,1]=(ori_w*(boxes[:, 1])+pad[0])/ padded_w
            boxes[:,2]=(ori_h*(boxes[:, 2])+pad[2])/ padded_h
            boxes[:,3]*=ori_w/padded_w
            boxes[:,4]*=ori_h/padded_h
            targets=torch.zeros((len(boxes), 6))
            targets[:,1:]=boxes
            return img,targets.type(torch.float32)
        else:
            return img,None
        
        
    
    def __len__(self):
        return self._len
        
    
    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))
        for ind,val in enumerate(targets):
            if val is not None:
                val[:,0]=ind 
        targets=[val for val in targets if val is not None]
        imgs=torch.stack(imgs,0)
        
        return imgs,targets
        