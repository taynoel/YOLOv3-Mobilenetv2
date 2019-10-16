import torch.nn as nn


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def _extendedLayers(inputChan,internalChan):
    return nn.Sequential(conv_1x1_bn(inputChan, internalChan),
                         nn.Conv2d(internalChan,internalChan,3,1,1,groups=internalChan, bias=False),
                         nn.BatchNorm2d(internalChan),
                         nn.ReLU6(inplace=True),
                         conv_1x1_bn(internalChan, internalChan*2),
    
                        conv_1x1_bn(internalChan*2, internalChan),
                        nn.Conv2d(internalChan,internalChan,3,1,1,groups=internalChan, bias=False),
                        nn.BatchNorm2d(internalChan),
                        nn.ReLU6(inplace=True),
                        conv_1x1_bn(internalChan, internalChan*2),
    
                        conv_1x1_bn(internalChan*2, internalChan))

def _outputLayers(internalChan,outputChan):
    return nn.Sequential(nn.Conv2d(internalChan,internalChan,3,1,1,groups=internalChan, bias=False),
                        nn.BatchNorm2d(internalChan),
                        nn.ReLU6(inplace=True),
                        conv_1x1_bn(internalChan, internalChan*2),
                        nn.Conv2d(internalChan*2,outputChan,1,1,0))
    
    
    

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio,kernel=3):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

