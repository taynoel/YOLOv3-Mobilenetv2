import cv2
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

def showImgT_WH(imgT,mode=1):
    if mode==0:
        cv2.imshow("im",imgT.cpu().numpy())
        cv2.waitKey(0)
    elif mode==1:
        plt.imshow(imgT.permute(1,2,0).cpu().numpy())

def showImgNLab(imgT,labT,mode=1,scale=None):
    if mode==0:
        pass
    elif mode==1:
        if scale is None:
            _,imgH,imgW=imgT.shape
        else:
            imgH,imgW=[1,1]
        imgT=imgT.clone()
        #img=imgT.permute(1,2,0).cpu().numpy()
        for fB in range(labT.shape[0]):
            x=labT[fB,2].item()
            y=labT[fB,3].item()
            w=labT[fB,4].item()
            h=labT[fB,5].item()
            
            leftX=int((x-w/2)*imgW)
            rightX=int((x+w/2)*imgW)
            upY=int((y-h/2)*imgH)
            downY=int((y+h/2)*imgH)
            
            imgT[:,upY:upY+2,leftX:rightX].fill_(1.)
            imgT[:,downY:downY+2,leftX:rightX].fill_(1.)
            imgT[:,upY:downY,leftX:leftX+2].fill_(1.)
            imgT[:,upY:downY,rightX:rightX+2].fill_(1.)
        plt.imshow(imgT.permute(1,2,0).cpu().numpy())
            
def imgRead(fileName,imgSize):
        img = transforms.ToTensor()(Image.open(fileName).convert('RGB'))
        return imgTransformSingleImg(img,0,0.5,0.5,imgSize)[0]    
        

def frameIndicesForIntendedFps(startClip,numOfClips,intendedFrameRate, vidFrameRate):
    numVidFramePerCount=vidFrameRate/intendedFrameRate
    return [int(f*numVidFramePerCount+startClip) for f in range(numOfClips)]

def imgTransformSingleImg(img,flipFlag,padHeightRatio,padWidthRatio,squareSize):
    if flipFlag: 
        _frame=img.flip(2)
    else: 
        _frame=img
    maxDim=max(_frame.shape[1],_frame.shape[2])
    _frame=F.interpolate(_frame.unsqueeze(0),scale_factor=squareSize/maxDim,mode="nearest")[0]
    _,orih,oriw=_frame.shape
    _frame,pad = padImg2SquareArbitrary(_frame,padHeightRatio,padWidthRatio)
    _frame = ImgTSquareResize(_frame,squareSize)
    return _frame,pad,[orih,oriw]


def findOverlappedlength(left1,right1,left2,right2):
    return max(0,min(right1,right2)-max(left1,left2))

  

def padImg2Square(imgT):
    #receive torch tensor imgT.size (channel,height,width)
    _,hgt,wdt=imgT.shape
    padVal=np.abs(hgt-wdt)//2
    if hgt<wdt:
        pad = (0,0,int(padVal),int(padVal))
    else:
        pad = (int(padVal),int(padVal),0,0)
    imgPadded = F.pad(imgT, pad, "constant", value=0.)
    return imgPadded

def padImg2SquareArbitrary(imgT,hr,wr):
    #receive torch tensor imgT.size (channel,height,width)
    #pad=(left_pad,right_pad,upper_pad,lower_pad) absolute
    if len(imgT.shape)==3:
        _,hgt,wdt=imgT.shape
    elif len(imgT.shape)==4:
        _,_,hgt,wdt=imgT.shape
    padVal=np.abs(hgt-wdt)
    upperPad,lowerPad=hr*padVal,(1-hr)*padVal
    leftPad,rightPad=wr*padVal,(1-wr)*padVal
    if hgt<wdt:
        pad = (0,0,int(upperPad),int(lowerPad))
    else:
        pad = (int(leftPad),int(rightPad),0,0)
    imgPadded = F.pad(imgT, pad, "constant", value=0.)
    return imgPadded,pad
    
def ImgTSquareResize(imgT,size):
    #receive torch tensor imgT.size (channel,height,width)
    return F.interpolate(imgT.unsqueeze(0),size=size,mode="nearest")[0]


