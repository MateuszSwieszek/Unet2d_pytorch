import torch
import torch.nn as nn
import numpy as np

class Double_Conv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1,bias=False) -> None:
        super(Double_Conv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,out_channels=output_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias), # We don't use bias due to batch norm
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=output_channels,out_channels=output_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias), # We don't use bias due to batch norm
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNET(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, features=[64, 128, 256, 512]) -> None:
        super(UNET,self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(Double_Conv(input_channels, feature))
            input_channels = feature

        self.bottleneck = Double_Conv(features[-1],features[-1]*2)
        
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2,feature,kernel_size=2, stride=2))  #https://medium.com/apache-mxnet/transposed-convolutions-explained-with-ms-excel-52d13030c7e8
            self.ups.append(Double_Conv(feature*2,feature))
        self.final_conv = nn.Conv2d(features[0], output_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        return self.final_conv(x)