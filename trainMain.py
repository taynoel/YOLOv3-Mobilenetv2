"""
Img with no box label, how?
random img flip, target should also be flipped

Notes:
need to use leaky relu; no relu before yolo layer
"""


from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import argparse
import dataset
import model_sum as model

import matplotlib.pyplot as plt
import torch
import imgUtils
import utils

MODE="train"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainDatasetListFile', type=str, default="trainList.part",
                        help='training dataset list file')
    parser.add_argument('--trainDatasetDirectory', type=str, default="./dataset/img",
                        help='training dataset directory')
    parser.add_argument('--trainDatasetLabelDirectory', type=str, default="./dataset/label",
                        help='training dataset directory')
    
    parser.add_argument('--imgSquareSize', type=int, default=416,
                        help='Padded squared image size length')
    parser.add_argument('--batchSize', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--pretrainedParamFile', type=str, default="yoloParam.dict",
                        help='Pretrained parameter file')
    args = parser.parse_args()
    return args

if __name__=="__main__":
    options=parse_args()
    
    trainDataSet=dataset.ListDataset(options)
    device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
    net=model.objDetNet(options)
    net.to(device)
    net.loadPretrainedParams()
    
    if MODE is "train":
        dataloaderTrain = DataLoader(
                trainDataSet,
                batch_size=options.batchSize,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                collate_fn=trainDataSet.collate_fn
            )
        optimizer=optim.Adam(net.parameters(),lr=0.001,eps=1e-3)
        _trainCount=0
        for _ in range(10000):
            print("======New Epoch=======")
            for inputs,labels in dataloaderTrain:
                optimizer.zero_grad()
                inputs=Variable(inputs.to(device))
                labels=(Variable(torch.cat(labels,dim=0).to(device)) if labels!=[] else 
                        Variable(torch.FloatTensor(0,6).to(device)))
                out,loss=net(inputs,labels)
                print(_trainCount,loss.item())
                loss.backward()
                optimizer.step()
                _trainCount+=1
                if _trainCount%500==0 and _trainCount>10:
                        torch.save(net.state_dict(),"yoloParam%d.dict"%_trainCount)
    elif MODE is "predict":
        fileName='img.jpg'
        net.eval()
        img=Variable(trainDataSet.imgRead(fileName).unsqueeze(0).to(device))
        with torch.no_grad():
            out,_=net(img)
        pred=torch.cat(out,dim=1).cpu()
        detections = utils.non_max_suppression(pred, 0.6, 0.4)[0]
        a,label=torch.split(detections,[6,1],dim=1)
        label=torch.cat([torch.zeros(label.shape[0],1),label,a],dim=1)
        label[:,2:6]=utils.xyxy2xywh(label[:,2:6])/options.imgSquareSize
        imgUtils.showImgNLab(img[0],label)
        
    
    
                    
                    
            
        
    