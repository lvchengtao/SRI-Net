import torch
import torch.nn as nn
import torchvision.models as models
from ResNet import ResNet50
from torch.nn import functional as F
class Dilate(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Dilate,self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):

        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x=max_out

        x = self.conv1(x)
        return self.sigmoid(x)*x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):

        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)*x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
class Refine(nn.Module):
    def __init__(self,in_channels):
        super(Refine,self).__init__()
        self.conv = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)
    def forward(self,x):
        out = x+self.conv(x)
        return out
    
#SRINet
class SRINet(nn.Module):
    def __init__(self, channel=32):
        super(SRINet, self).__init__()
        
        # Backbone model
        self.resnet = ResNet50('rgb')
        self.resnet_depth=ResNet50('rgb')

        #Decoder 1
        self.DI1_0 = Dilate(64,64)
        self.DI1_1 = Dilate(256,64)
        self.DI1_4 = Dilate(2048,64)
        self.CA1_0 = ChannelAttention(128)
        self.CA1_1 = ChannelAttention(128)
        self.conv3x3_1_0 = nn.Conv2d(128,64,kernel_size=3,padding=1)
        self.ba1_0 = nn.BatchNorm2d(64)
        self.relu1_0 = nn.ReLU()
        self.conv3x3_1_1 = nn.Conv2d(128,64,kernel_size=3,padding=1)
        self.ba1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU()
        self.conv3x3_1 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.conv_out_1 = nn.Conv2d(64,1,kernel_size=1)
        

        #upsample function
        self.upsample6 = nn.Upsample(scale_factor=64, mode='bilinear', align_corners=True)
        self.upsample5 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.DI2_0 = BasicConv2d(64, 64, 3, padding=1)
        self.DI2_1 = BasicConv2d(256, 64, 3, padding=1)
        self.DI2_2 = BasicConv2d(512, 64, 3, padding=1)
        self.DI2_3 = BasicConv2d(1024, 64, 3, padding=1)
        self.DI2_4 = BasicConv2d(2048, 64, 3, padding=1)
        
        self.conv_ca0 = nn.Conv2d(128,64,kernel_size=3,padding=1)
        self.ba_ca0 = nn.BatchNorm2d(64)
        self.relu_ca0 = nn.ReLU()

        self.conv_ca1 = nn.Conv2d(128,64,kernel_size=3,padding=1)
        self.ba_ca1 = nn.BatchNorm2d(64)
        self.relu_ca1 = nn.ReLU()

        self.conv_ca2 = nn.Conv2d(128,64,kernel_size=3,padding=1)
        self.ba_ca2 = nn.BatchNorm2d(64)
        self.relu_ca2 = nn.ReLU()

        self.conv_ca3 = nn.Conv2d(128,64,kernel_size=3,padding=1)
        self.ba_ca3 = nn.BatchNorm2d(64)
        self.relu_ca3 = nn.ReLU()
        #refine   
        self.ref = Refine(64)
        #out
        self.conv_sal_re = nn.Conv2d(64,1,kernel_size=3,padding=1)
        self.conv_sal0 = nn.Conv2d(64,1,kernel_size=3,padding=1)
        self.conv_sal1 = nn.Conv2d(64,1,kernel_size=3,padding=1)
        self.conv_sal2 = nn.Conv2d(64,1,kernel_size=3,padding=1)
        self.conv_sal3 = nn.Conv2d(64,1,kernel_size=3,padding=1)
        self.conv_sal4 = nn.Conv2d(64,1,kernel_size=3,padding=1)

        if self.training:
            self.initialize_weights()

    def forward(self, x, x_depth):
        bz= x.shape[0]
        x_o= x

        x_depth_o= x_depth.unsqueeze(dim=1)

        x_depth_o= torch.cat(torch.chunk(x_depth_o, 12, dim=2), dim=1)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)  # 256 x 64 x 64

        x2 = self.resnet.layer2(x1)  # 512 x 32 x 32

        x2_1 = x2

        x3_1 = self.resnet.layer3_1(x2_1)  # 1024 x 16 x 16

        x4_1 = self.resnet.layer4_1(x3_1)  # 2048 x 8 x 8
        
        x1_4 = self.DI1_4(x4_1)
        x1_1 = self.DI1_1(x1)
        x1_0 = self.DI1_0(x)
        f1_1 = self.relu1_1(self.ba1_1(self.conv3x3_1_1(self.CA1_1(torch.cat((x1_1,self.upsample3(x1_4)),1)))))
        f1_0 = self.relu1_0(self.ba1_0(self.conv3x3_1_0(self.CA1_0(torch.cat((x1_0,f1_1),1)))))
        s2_x_depth_1 = self.upsample2(self.conv_out_1(self.conv3x3_1(f1_0)))

        s2_x_depth_1_w = torch.sigmoid(s2_x_depth_1)


        image= torch.mul(x_o, s2_x_depth_1_w)
        depth = torch.mul(x_depth, s2_x_depth_1_w)
       
        depth1, depth2, depth3, depth4, depth5, depth6, depth7, depth8, depth9, depth10, depth11, depth12= torch.chunk(depth, 12, dim=1) 
        mae1= torch.mean(torch.abs(torch.sub(image, depth1)), dim=[1,2,3])
        mae2= torch.mean(torch.abs(torch.sub(image, depth2)), dim=[1,2,3])
        mae3= torch.mean(torch.abs(torch.sub(image, depth3)), dim=[1,2,3])
        mae4= torch.mean(torch.abs(torch.sub(image, depth4)), dim=[1,2,3])
        mae5= torch.mean(torch.abs(torch.sub(image, depth5)), dim=[1,2,3])
        mae6= torch.mean(torch.abs(torch.sub(image, depth6)), dim=[1,2,3])
        mae7= torch.mean(torch.abs(torch.sub(image, depth7)), dim=[1,2,3])
        mae8= torch.mean(torch.abs(torch.sub(image, depth8)), dim=[1,2,3])
        mae9= torch.mean(torch.abs(torch.sub(image, depth9)), dim=[1,2,3])
        mae10= torch.mean(torch.abs(torch.sub(image, depth10)), dim=[1,2,3])
        mae11= torch.mean(torch.abs(torch.sub(image, depth11)), dim=[1,2,3])
        mae12= torch.mean(torch.abs(torch.sub(image, depth12)), dim=[1,2,3])


        min_v, min_i= torch.min(torch.cat([mae1.unsqueeze(dim=1), mae2.unsqueeze(dim=1), mae3.unsqueeze(dim=1), mae4.unsqueeze(dim=1), mae5.unsqueeze(dim=1), mae6.unsqueeze(dim=1), mae7.unsqueeze(dim=1), mae8.unsqueeze(dim=1), mae9.unsqueeze(dim=1), mae10.unsqueeze(dim=1), mae11.unsqueeze(dim=1), mae12.unsqueeze(dim=1)], dim=1), dim=1)

        
        for i in range(bz):
            if i == 0:
                depth_n= x_depth_o[i, min_i[i]].unsqueeze(dim=0)
            else:
                depth_n= torch.cat([depth_n, x_depth_o[i, min_i[i]].unsqueeze(dim=0)], dim=0)
        
        with torch.no_grad():
            depth_n= depth_n
        x = self.resnet.conv1(x_o)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)

        x = self.resnet.maxpool(x)

        x_depth = self.resnet_depth.conv1(depth_n)
        x_depth = self.resnet_depth.bn1(x_depth)
        x_depth = self.resnet_depth.relu(x_depth)
        x_depth = self.resnet_depth.maxpool(x_depth)

        #layer0 merge,
        f0 = x+x_depth
        #layer0 merge end

        x1 = self.resnet.layer1(x)  # 256 x 64 x 64
        x1_depth=self.resnet_depth.layer1(x_depth)

        #layer1 merge

        f1 = x1+x1_depth
        #layer1 merge end

        x2 = self.resnet.layer2(x1)  # 512 x 32 x 32
        x2_depth=self.resnet_depth.layer2(x1_depth)

        #layer2 merge
        f2 = x2+x2_depth
        #layer2 merge end

        x2_1 = x2

        x3_1 = self.resnet.layer3_1(x2_1)  # 1024 x 16 x 16
        x3_1_depth=self.resnet_depth.layer3_1(x2_depth)

        #layer3_1 merge
        f3 = x3_1+x3_1_depth
        #layer3_1 merge end

        x4_1 = self.resnet.layer4_1(x3_1)  # 2048 x 8 x 8
        x4_1_depth=self.resnet_depth.layer4_1(x3_1_depth)

        #layer4_1 merge
        f4 = x4_1+x4_1_depth
        #layer4_1 merge end
        
        #produce initial saliency map by decoder1
        f4 = self.DI2_4(f4)
        f3 = self.DI2_3(f3)
        f2 = self.DI2_2(f2)
        f1 = self.DI2_1(f1)
        f0 = self.DI2_0(f0)

        C4 = f4
        C3 = self.relu_ca3(self.ba_ca3(self.conv_ca3(torch.cat((self.upsample1(C4),f3),1))))
        C2 = self.relu_ca2(self.ba_ca2(self.conv_ca2(torch.cat((self.upsample1(C3),f2),1))))

        C1 = self.relu_ca1(self.ba_ca1(self.conv_ca1(torch.cat((self.upsample1(C2),f1),1))))
        C0 = self.relu_ca0(self.ba_ca0(self.conv_ca0(torch.cat((C1,f0),1))))

        S_re = self.ref(C0)
        sal_re = self.upsample2(self.conv_sal_re(S_re))
        sal0 = self.upsample2(self.conv_sal0(C0))
        sal1 = self.upsample2(self.conv_sal1(C1))
        sal2 = self.upsample3(self.conv_sal2(C2))
        sal3 = self.upsample4(self.conv_sal3(C3))
        sal4 = self.upsample5(self.conv_sal4(C4))

        return F.sigmoid(sal_re), F.sigmoid(sal0), F.sigmoid(sal1),F.sigmoid(sal2),F.sigmoid(sal3), F.sigmoid(sal4),s2_x_depth_1_w
    
    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)
    
    #initialize the weights
    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)

        all_params = {}
        for k, v in self.resnet_depth.state_dict().items():
            # if k=='conv1.weight':
            #     all_params[k]=torch.nn.init.normal_(v, mean=0, std=1)
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet_depth.state_dict().keys())
        self.resnet_depth.load_state_dict(all_params)
if __name__ == '__main__':
    pass

