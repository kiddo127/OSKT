import math
import torch
from torch import nn
from torch.nn import functional as F
import json




#region new network coupling

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, last_stride=2, block=Bottleneck, layers=[3, 4, 6, 3], in_planes=64, multipliers=[2,4,8]):
        super().__init__()

        self.in_planes = in_planes
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, in_planes, layers[0])
        self.layer2 = self._make_layer(block, in_planes*multipliers[0], layers[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes*multipliers[1], layers[2], stride=2)
        self.layer4 = self._make_layer(block, in_planes*multipliers[2], layers[3], stride=last_stride)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = nn.Sequential(
            nn.Conv2d(self.in_planes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

#endregion







#region IBN-Net

class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = planes//2
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)
    
    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out

class Bottleneck_IBN(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, ibn=False, stride=1, downsample=None):
        super(Bottleneck_IBN, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        if ibn:
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.gene_match_idx = {}
        self.ibn = ibn

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet_IBN(nn.Module):

    def __init__(self, last_stride, block=Bottleneck_IBN, layers=[3,4,6,3], in_planes=64, multipliers=[2,4,8]):
        super(ResNet_IBN, self).__init__()
        self.in_planes = in_planes
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        # self.bn1 = IBN(in_planes)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=None, padding=0)

        self.gene_match_idx = {}

        self.layer1 = self._make_layer(block, in_planes, layers[0])
        self.layer2 = self._make_layer(block, in_planes*multipliers[0], layers[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes*multipliers[1], layers[2], stride=2)
        self.layer4 = self._make_layer(block, in_planes*multipliers[2], layers[3], stride=last_stride, ibn=False)
        # self.avgpool = nn.AvgPool2d(7)
        # self.fc = nn.Linear(in_planes * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, ibn=True):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, ibn, stride, downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, ibn))

        return nn.Sequential(*layers)

    def load_gene_matcher(self,load_path):
        summary_dict = json.load(open(load_path))
        self.gene_match_idx['conv1'] = summary_dict['base.conv1']

        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for layer_idx, layer in enumerate(layers):
            for block_idx, block in enumerate(layer):
                block.gene_match_idx['conv1'] = summary_dict['base.layer'+str(layer_idx+1)+'.'+str(block_idx)+'.conv1']
                if block.ibn:
                    block.gene_match_idx['bn1.IN'] = summary_dict['base.layer'+str(layer_idx+1)+'.'+str(block_idx)+'.bn1.IN']
                    block.gene_match_idx['bn1.BN'] = summary_dict['base.layer'+str(layer_idx+1)+'.'+str(block_idx)+'.bn1.BN']
                block.gene_match_idx['conv2'] = summary_dict['base.layer'+str(layer_idx+1)+'.'+str(block_idx)+'.conv2']
                block.gene_match_idx['conv3'] = summary_dict['base.layer'+str(layer_idx+1)+'.'+str(block_idx)+'.conv3']
                if block.downsample is not None:
                    block.gene_match_idx['downsample'] = summary_dict['base.layer'+str(layer_idx+1)+'.'+str(block_idx)+'.downsample']

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            try:
                self.state_dict()[i].copy_(param_dict[i])
            except Exception as e:
                print(i)
                # planes = param_dict[i].shape[0]
                # if 'bias' in i or 'weight' in i:
                #     self.state_dict()[i.replace('bn1','bn1.IN')].copy_(param_dict[i][:planes//2])
                # self.state_dict()[i.replace('bn1','bn1.BN')].copy_(param_dict[i][planes//2:])

#endregion

