import math
import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
from torch.nn import init
from scipy.spatial import distance
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
from collections import Counter
import copy
import random



def find_all_indices(lst, label):
    return [index for index, value in enumerate(lst) if value == label]

def select_elements(A, m):
    # to_gene: for current layer
    gene_num = len(set(A))
    assert gene_num <= m

    A_count = Counter(A)
    assert gene_num == len(A_count) == (max(A)+1)

    gene_count = {label:1 for label in range(gene_num)}
    to_gene = [i for i in range(gene_num)]

    while len(to_gene) < m:
        ratios = [gene_count[i]/A_count[i] for i in range(gene_num)]
        min_index = ratios.index(min(ratios))
        to_gene.append(min_index)
        gene_count[min_index] += 1
    
    assert len(to_gene) == m

    # to_origin: for next layer
    to_origin = {}

    for label in range(gene_num):
        g2o = find_all_indices(A, label)
        idxs = find_all_indices(to_gene, label)
        chushu = len(g2o) // len(idxs)
        yushu = len(g2o) % len(idxs)
        pos = 0
        for idx in idxs:
            if yushu:
                to_origin[idx] = g2o[pos:pos+chushu+1]
                pos = pos+chushu+1
                yushu = yushu-1
            else:
                to_origin[idx] = g2o[pos:pos+chushu]
                pos = pos+chushu
        assert pos == len(g2o)
        assert yushu == 0
    assert len(to_gene) == len(to_origin)

    return to_gene, to_origin

def random_select_with_duplicates(M, N):
    numbers = list(range(M))
    labels = list(range(M))
    if len(labels) < N:
        labels += random.choices(numbers,k=(N-len(labels)))

    assert len(labels) == N

    return labels



class Gene_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, genenet_planes=None):
        super(Gene_Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, genenet_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, genenet_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, genenet_planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.gene_match_idx = {}

        self.trans_conv1 = nn.Conv2d(in_channels=genenet_planes, out_channels=planes, kernel_size=1, stride=1, bias=False)
        self.trans_bn1 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=1, stride=1, bias=False)
        self.trans_conv2 = nn.Conv2d(in_channels=genenet_planes, out_channels=planes, kernel_size=1, stride=1, bias=False)
        self.trans_bn2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=1, stride=1, bias=False)
        self.trans_conv3 = nn.Conv2d(in_channels=genenet_planes*4, out_channels=planes*4, kernel_size=1, stride=1, bias=False)
        self.trans_bn3 = nn.Conv2d(in_channels=planes*4, out_channels=planes*4, kernel_size=1, stride=1, bias=False)
        if downsample is not None:
            self.trans_downsample0 = nn.Conv2d(in_channels=genenet_planes*4, out_channels=planes*4, kernel_size=1, stride=1, bias=False)
            self.trans_downsample1 = nn.Conv2d(in_channels=planes*4, out_channels=planes*4, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.trans_conv1(out)
        out = self.bn1(out)
        out = self.trans_bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.trans_conv2(out)
        out = self.bn2(out)
        out = self.trans_bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.trans_conv3(out)
        out = self.bn3(out)
        out = self.trans_bn3(out)

        if self.downsample is not None:
            residual = self.downsample[0](x)
            residual = self.trans_downsample0(residual)
            residual = self.downsample[1](residual)
            residual = self.trans_downsample1(residual)

        out += residual
        out = self.relu(out)

        return out

class GeneChain(nn.Module):
    def __init__(self, last_stride=2, block=Gene_Bottleneck, layers=[3, 4, 6, 3], in_planes=64, multipliers=[2,4,8],
                 weight_chain_in_planes=None):
        super().__init__()

        self.in_planes = in_planes
        self.gene_match_idx = {}
        self.GeneChain_feat_idx = []

        self.weight_chain_in_planes = weight_chain_in_planes
        self.conv1 = nn.Conv2d(3, self.weight_chain_in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.trans_conv1 = nn.Conv2d(in_channels=weight_chain_in_planes, out_channels=in_planes, kernel_size=1, stride=1, bias=False)
        self.trans_bn1 = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=1, stride=1, bias=False) # trans参数太大了 看看能不能减掉 组卷积或者什么的
        # self.trans_conv_setting()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, in_planes, layers[0], stride=1, genenet_planes=weight_chain_in_planes)
        self.layer2 = self._make_layer(block, in_planes*multipliers[0], layers[1], stride=2, genenet_planes=weight_chain_in_planes*multipliers[0])
        self.layer3 = self._make_layer(block, in_planes*multipliers[1], layers[2], stride=2, genenet_planes=weight_chain_in_planes*multipliers[1])
        self.layer4 = self._make_layer(block, in_planes*multipliers[2], layers[3], stride=last_stride, genenet_planes=weight_chain_in_planes*multipliers[2]) #使用last stride=1，提高分辨率

    def _make_layer(self, block, planes, blocks, stride=1, genenet_planes=None):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, genenet_planes*block.expansion,
                          kernel_size=1, stride=stride, bias=False), #layer1中的downsample分辨率并没有降采样,stride=1,但是通道数变为planes*block.expansion
                nn.BatchNorm2d(planes*block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, genenet_planes=genenet_planes, ))
        self.in_planes = planes * block.expansion
        self.weight_chain_in_planes = genenet_planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, genenet_planes=genenet_planes, ))

        return nn.Sequential(*layers)

    def trans_conv_setting(self):
        self.trans_conv1.weight.requires_grad = False
        self.trans_conv1.weight.zero_()
        self.trans_bn1.weight.requires_grad = False
        self.trans_bn1.weight.zero_()
        for g_idx, o_idxs in self.gene_match_idx['conv1'].items():
            for o_idx in o_idxs:
                self.trans_conv1.weight[o_idx, g_idx, 0, 0] = 1
                self.trans_bn1.weight[o_idx, o_idxs, 0, 0] = 1.0/len(o_idxs)

        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for layer_idx, layer in enumerate(layers):
            for block_idx, block in enumerate(layer):
                block.trans_conv1.weight.requires_grad = False
                block.trans_conv1.weight.zero_()
                block.trans_bn1.weight.requires_grad = False
                block.trans_bn1.weight.zero_()
                for g_idx, o_idxs in block.gene_match_idx['conv1'].items():
                    for o_idx in o_idxs:
                        block.trans_conv1.weight[o_idx, g_idx, 0, 0] = 1
                        block.trans_bn1.weight[o_idx, o_idxs, 0, 0] = 1.0/len(o_idxs)

                block.trans_conv2.weight.requires_grad = False
                block.trans_conv2.weight.zero_()
                block.trans_bn2.weight.requires_grad = False
                block.trans_bn2.weight.zero_()
                for g_idx, o_idxs in block.gene_match_idx['conv2'].items():
                    for o_idx in o_idxs:
                        block.trans_conv2.weight[o_idx, g_idx, 0, 0] = 1
                        block.trans_bn2.weight[o_idx, o_idxs, 0, 0] = 1.0/len(o_idxs)

                block.trans_conv3.weight.requires_grad = False
                block.trans_conv3.weight.zero_()
                block.trans_bn3.weight.requires_grad = False
                block.trans_bn3.weight.zero_()
                for g_idx, o_idxs in block.gene_match_idx['conv3'].items():
                    for o_idx in o_idxs:
                        block.trans_conv3.weight[o_idx, g_idx, 0, 0] = 1
                        block.trans_bn3.weight[o_idx, o_idxs, 0, 0] = 1.0/len(o_idxs)

                if block.downsample is not None:
                    block.trans_downsample0.weight.requires_grad = False
                    block.trans_downsample0.weight.zero_()
                    block.trans_downsample1.weight.requires_grad = False
                    block.trans_downsample1.weight.zero_()
                    for g_idx, o_idxs in block.gene_match_idx['downsample'].items():
                        for o_idx in o_idxs:
                            block.trans_downsample0.weight[o_idx, g_idx, 0, 0] = 1
                            block.trans_downsample1.weight[o_idx, o_idxs, 0, 0] = 1.0/len(o_idxs)
        
        for g_idx, o_idxs in self.layer4[-1].gene_match_idx['conv3'].items():
            self.GeneChain_feat_idx.append(o_idxs[0])

    def forward(self,x):
        x = self.conv1(x)
        x = self.trans_conv1(x)
        x = self.bn1(x)
        x = self.trans_bn1(x)
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
            if 'classifier' in i or 'bottleneck' in i:
                continue
            self.state_dict()[i.replace('base.','')].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class Bottleneck_wGene(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, Gene_Bottleneck, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.Gene_Bottleneck = Gene_Bottleneck
        self.gene_match_idx = {}


    def get_refining_loss(self):
        # ============================1============================
        filter_num = self.conv1.weight.shape[0]
        res = (self.conv1.weight-self.Gene_Bottleneck.conv1.weight[self.gene_match_idx['conv1']]).view(filter_num,-1)
        loss = (res*res).sum()

        # ============================2============================
        filter_num = self.conv2.weight.shape[0]
        res = (self.conv2.weight-self.Gene_Bottleneck.conv2.weight[self.gene_match_idx['conv2']]).view(filter_num,-1)
        loss += (res*res).sum()

        # ============================3============================
        filter_num = self.conv3.weight.shape[0]
        res = (self.conv3.weight-self.Gene_Bottleneck.conv3.weight[self.gene_match_idx['conv3']]).view(filter_num,-1)
        loss += (res*res).sum()

        # ============================downsample============================
        if self.downsample is not None:
            filter_num = self.downsample[0].weight.shape[0]
            res = (self.downsample[0].weight-self.Gene_Bottleneck.downsample[0].weight[self.gene_match_idx['downsample']]).view(filter_num,-1)
            loss += (res*res).sum()

        return loss

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

class ResNet_wGene(nn.Module):
    def __init__(self, last_stride=2, block=Bottleneck_wGene, layers=[3, 4, 6, 3], in_planes=64, multipliers=[2,4,8],
                 weight_chain_in_planes=None):
        super().__init__()

        self.in_planes = in_planes
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.weight_chain_in_planes = weight_chain_in_planes
        self.GeneChain = GeneChain(last_stride=last_stride,in_planes=in_planes,multipliers=multipliers,
                                   weight_chain_in_planes=weight_chain_in_planes)
        self.isGrowing = False
        self.gene_match_idx = {}

        GeneChainL1 = getattr(self.GeneChain,'layer1')
        GeneChainL2 = getattr(self.GeneChain,'layer2')
        GeneChainL3 = getattr(self.GeneChain,'layer3')
        GeneChainL4 = getattr(self.GeneChain,'layer4')

        self.layer1 = self._make_layer(block, in_planes, layers[0], GeneChainL1, stride=1)
        self.layer2 = self._make_layer(block, in_planes*multipliers[0], layers[1], GeneChainL2, stride=2)
        self.layer3 = self._make_layer(block, in_planes*multipliers[1], layers[2], GeneChainL3, stride=2)
        self.layer4 = self._make_layer(block, in_planes*multipliers[2], layers[3], GeneChainL4, stride=last_stride)

    def _make_layer(self, block, planes, blocks, Gene_Bottlenecks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        tmp_blk = block(self.in_planes, planes, Gene_Bottlenecks[0], stride, downsample)
        layers.append(tmp_blk)
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            tmp_blk = block(self.in_planes, planes, Gene_Bottlenecks[i])
            layers.append(tmp_blk)

        return nn.Sequential(*layers)

    def save_gene_matcher(self,save_path):
        summary_dict = {}

        summary_dict['base.conv1'] = self.gene_match_idx['conv1']
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for layer_idx, layer in enumerate(layers):
            for block_idx, block in enumerate(layer):
                summary_dict['base.layer'+str(layer_idx+1)+'.'+str(block_idx)+'.conv1'] = block.gene_match_idx['conv1']
                summary_dict['base.layer'+str(layer_idx+1)+'.'+str(block_idx)+'.conv2'] = block.gene_match_idx['conv2']
                summary_dict['base.layer'+str(layer_idx+1)+'.'+str(block_idx)+'.conv3'] = block.gene_match_idx['conv3']
                if block.downsample is not None:
                    summary_dict['base.layer'+str(layer_idx+1)+'.'+str(block_idx)+'.downsample'] = block.gene_match_idx['downsample']

        json.dump(summary_dict, open(save_path,'w'))

    def load_gene_matcher(self,load_path):
        summary_dict = json.load(open(load_path))

        self.gene_match_idx['conv1'] = summary_dict['base.conv1']
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for layer_idx, layer in enumerate(layers):
            for block_idx, block in enumerate(layer):
                block.gene_match_idx['conv1'] = summary_dict['base.layer'+str(layer_idx+1)+'.'+str(block_idx)+'.conv1']
                block.gene_match_idx['conv2'] = summary_dict['base.layer'+str(layer_idx+1)+'.'+str(block_idx)+'.conv2']
                block.gene_match_idx['conv3'] = summary_dict['base.layer'+str(layer_idx+1)+'.'+str(block_idx)+'.conv3']
                if block.downsample is not None:
                    block.gene_match_idx['downsample'] = summary_dict['base.layer'+str(layer_idx+1)+'.'+str(block_idx)+'.downsample']

        if self.GeneChain is not None:
            cur_matcher = self.gene_match_idx['conv1']
            self.GeneChain.gene_match_idx['conv1'] = {label: find_all_indices(cur_matcher, label) for label in range(self.GeneChain.conv1.weight.shape[0])}
            layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            for layer in layers:
                for block in layer:
                    cur_matcher = block.gene_match_idx['conv1']
                    block.Gene_Bottleneck.gene_match_idx['conv1'] = {label: find_all_indices(cur_matcher, label) for label in range(block.Gene_Bottleneck.conv1.weight.shape[0])}
                    cur_matcher = block.gene_match_idx['conv2']
                    block.Gene_Bottleneck.gene_match_idx['conv2'] = {label: find_all_indices(cur_matcher, label) for label in range(block.Gene_Bottleneck.conv2.weight.shape[0])}
                    cur_matcher = block.gene_match_idx['conv3']
                    block.Gene_Bottleneck.gene_match_idx['conv3'] = {label: find_all_indices(cur_matcher, label) for label in range(block.Gene_Bottleneck.conv3.weight.shape[0])}
                    if block.downsample is not None:
                        cur_matcher = block.gene_match_idx['downsample']
                        block.Gene_Bottleneck.gene_match_idx['downsample'] = {label: find_all_indices(cur_matcher, label) for label in range(block.Gene_Bottleneck.downsample[0].weight.shape[0])}

    def ClusterCenter_GeneInit(self):
        for k,v in self.GeneChain.gene_match_idx['conv1'].items():
            self.GeneChain.state_dict()['conv1.weight'][k].copy_(self.conv1.weight.data[v].mean(dim=0))

        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for layer_idx,layer in enumerate(layers):
            for block_idx,block in enumerate(layer):
                print("Cluster Center Initializing Gene layer {}.{}".format(layer_idx+1,block_idx))
                # 1
                for k,v in block.Gene_Bottleneck.gene_match_idx['conv1'].items():
                    block.Gene_Bottleneck.state_dict()['conv1.weight'][k].copy_(block.conv1.weight.data[v].mean(dim=0))
                # 2
                for k,v in block.Gene_Bottleneck.gene_match_idx['conv2'].items():
                    block.Gene_Bottleneck.state_dict()['conv2.weight'][k].copy_(block.conv2.weight.data[v].mean(dim=0))
                # 3
                for k,v in block.Gene_Bottleneck.gene_match_idx['conv3'].items():
                    block.Gene_Bottleneck.state_dict()['conv3.weight'][k].copy_(block.conv3.weight.data[v].mean(dim=0))
                # downsample
                if block.downsample is not None:
                    for k,v in block.Gene_Bottleneck.gene_match_idx['downsample'].items():
                        block.Gene_Bottleneck.state_dict()['downsample.0.weight'][k].copy_(block.downsample[0].weight.data[v].mean(dim=0))

    def FilterClustering(self):
        filter_weight = np.array(self.conv1.weight.data.view(self.conv1.weight.shape[0],-1))
        n_clusters = self.GeneChain.conv1.weight.shape[0] # gene filter num
        kmeans = AgglomerativeClustering(n_clusters=n_clusters)
        labels = kmeans.fit_predict(filter_weight).tolist()
        # # wo_GInit
        assert len(set(labels)) == n_clusters
        self.gene_match_idx['conv1'] = labels
        self.GeneChain.gene_match_idx['conv1'] = {label: find_all_indices(labels, label) for label in range(n_clusters)}

        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for layer_idx,layer in enumerate(layers):
            downsample_weight = np.array(layer[0].downsample[0].weight.data.view(layer[0].downsample[0].weight.shape[0],-1))
            C3DS_filter_weights = downsample_weight
            for block_idx,block in enumerate(layer):
                print("Clustering layer {}.{}".format(layer_idx+1,block_idx))
                # conv1
                filter_weight = np.array(block.conv1.weight.data.view(block.conv1.weight.shape[0],-1))
                n_clusters = block.Gene_Bottleneck.conv1.weight.shape[0] # gene filter num
                kmeans = AgglomerativeClustering(n_clusters=n_clusters)
                labels = kmeans.fit_predict(filter_weight).tolist()
                # # wo_GInit
                assert len(set(labels)) == n_clusters
                block.gene_match_idx['conv1'] = labels
                block.Gene_Bottleneck.gene_match_idx['conv1'] = {label: find_all_indices(labels, label) for label in range(n_clusters)}
                # conv2
                filter_weight = np.array(block.conv2.weight.data.view(block.conv2.weight.shape[0],-1))
                n_clusters = block.Gene_Bottleneck.conv2.weight.shape[0] # gene filter num
                kmeans = AgglomerativeClustering(n_clusters=n_clusters)
                labels = kmeans.fit_predict(filter_weight).tolist()
                # # wo_GInit
                assert len(set(labels)) == n_clusters
                block.gene_match_idx['conv2'] = labels
                block.Gene_Bottleneck.gene_match_idx['conv2'] = {label: find_all_indices(labels, label) for label in range(n_clusters)}
                # conv3
                conv3_weight = np.array(block.conv3.weight.data.view(block.conv3.weight.shape[0],-1))
                C3DS_filter_weights = np.hstack((C3DS_filter_weights, conv3_weight))

            n_clusters = layer[0].Gene_Bottleneck.downsample[0].weight.shape[0] # gene filter num
            kmeans = AgglomerativeClustering(n_clusters=n_clusters)
            labels = kmeans.fit_predict(C3DS_filter_weights).tolist()
            # # wo_GInit
            # labels = random_select_with_duplicates(n_clusters,C3DS_filter_weights.shape[0])
            assert len(set(labels)) == n_clusters
            layer[0].gene_match_idx['downsample'] = labels
            layer[0].Gene_Bottleneck.gene_match_idx['downsample'] = {label: find_all_indices(labels, label) for label in range(n_clusters)}
            for block in layer:
                block.gene_match_idx['conv3'] = labels
                block.Gene_Bottleneck.gene_match_idx['conv3'] = {label: find_all_indices(labels, label) for label in range(n_clusters)}

    def get_refining_loss(self):
        filter_num = self.conv1.weight.shape[0]
        res = (self.conv1.weight-self.GeneChain.conv1.weight[self.gene_match_idx['conv1']]).view(filter_num,-1)
        loss = (res*res).sum()

        for block in self.layer1:
            loss += block.get_refining_loss()
        for block in self.layer2:
            loss += block.get_refining_loss()
        for block in self.layer3:
            loss += block.get_refining_loss()
        for block in self.layer4:
            loss += block.get_refining_loss()
        
        return loss

    def GeneChain_BN_coupling(self):
        self.GeneChain.bn1.weight.detach_()
        self.GeneChain.bn1.bias.detach_()
        self.GeneChain.bn1.weight.copy_(self.bn1.weight)
        self.GeneChain.bn1.bias.copy_(self.bn1.bias)

        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for layer in layers:
            for block in layer:
                # 1
                block.Gene_Bottleneck.bn1.weight.detach_()
                block.Gene_Bottleneck.bn1.bias.detach_()
                block.Gene_Bottleneck.bn1.weight.copy_(block.bn1.weight)
                block.Gene_Bottleneck.bn1.bias.copy_(block.bn1.bias)
                # 2
                block.Gene_Bottleneck.bn2.weight.detach_()
                block.Gene_Bottleneck.bn2.bias.detach_()
                block.Gene_Bottleneck.bn2.weight.copy_(block.bn2.weight)
                block.Gene_Bottleneck.bn2.bias.copy_(block.bn2.bias)
                # 3
                block.Gene_Bottleneck.bn3.weight.detach_()
                block.Gene_Bottleneck.bn3.bias.detach_()
                block.Gene_Bottleneck.bn3.weight.copy_(block.bn3.weight)
                block.Gene_Bottleneck.bn3.bias.copy_(block.bn3.bias)
                # downsample
                if block.downsample is not None:
                    block.Gene_Bottleneck.downsample[1].weight.detach_()
                    block.Gene_Bottleneck.downsample[1].bias.detach_()
                    block.Gene_Bottleneck.downsample[1].weight.copy_(block.downsample[1].weight)
                    block.Gene_Bottleneck.downsample[1].bias.copy_(block.downsample[1].bias)

    def forward(self, x):
        self.GeneChain_BN_coupling()
        GeneNet_out = self.GeneChain(x)[:,self.GeneChain.GeneChain_feat_idx]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x,GeneNet_out

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



#region new network coupling

class CoupledBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(CoupledBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.gene_match_idx = {}

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

class CoupledResNet(nn.Module):
    def __init__(self,last_stride=1,block=CoupledBottleneck,layers=[3, 4, 6, 3],in_planes=64,multipliers=[2,4,8],
                 mid_stride=2):
        super().__init__()

        self.gene_match_idx = {}

        self.in_planes = in_planes
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, in_planes, layers[0])
        self.layer2 = self._make_layer(block, int(in_planes*multipliers[0]), layers[1], stride=mid_stride)
        self.layer3 = self._make_layer(block, int(in_planes*multipliers[1]), layers[2], stride=mid_stride)
        self.layer4 = self._make_layer(block, int(in_planes*multipliers[2]), layers[3], stride=last_stride)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = nn.Sequential(
            nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def load_gene_matcher(self,load_path):
        summary_dict = json.load(open(load_path))

        self.gene_match_idx['conv1'] = summary_dict['base.conv1']
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for layer_idx, layer in enumerate(layers):
            for block_idx, block in enumerate(layer):
                block.gene_match_idx['conv1'] = summary_dict['base.layer'+str(layer_idx+1)+'.'+str(block_idx)+'.conv1']
                block.gene_match_idx['conv2'] = summary_dict['base.layer'+str(layer_idx+1)+'.'+str(block_idx)+'.conv2']
                block.gene_match_idx['conv3'] = summary_dict['base.layer'+str(layer_idx+1)+'.'+str(block_idx)+'.conv3']
                if block.downsample is not None:
                    block.gene_match_idx['downsample'] = summary_dict['base.layer'+str(layer_idx+1)+'.'+str(block_idx)+'.downsample']

        cur_dict = self.gene_match_idx['conv1']
        filter_num = self.conv1.weight.shape[0]
        self.gene_match_idx['conv1_2g'],self.gene_match_idx['conv1_2o'] = select_elements(cur_dict,filter_num)

        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for layer_idx, layer in enumerate(layers):
            for block_idx, block in enumerate(layer):
                for conv_name in ['conv1','conv2','conv3']:
                    cur_dict = block.gene_match_idx[conv_name]
                    filter_num = block.__getattr__(conv_name).weight.shape[0]
                    block.gene_match_idx[conv_name+'_2g'],block.gene_match_idx[conv_name+'_2o'] = select_elements(cur_dict,filter_num)
                if block.downsample is not None:
                    cur_dict = block.gene_match_idx['downsample']
                    filter_num = block.downsample[0].weight.shape[0]
                    block.gene_match_idx['downsample_2g'],block.gene_match_idx['downsample_2o'] = select_elements(cur_dict,filter_num)
                assert layer[0].gene_match_idx['downsample_2g'] == block.gene_match_idx['conv3_2g']
                assert layer[0].gene_match_idx['downsample_2o'] == block.gene_match_idx['conv3_2o']


    def Init_via_Gene(self, pretrain_path, gene_matcher_path):
        self.load_gene_matcher(gene_matcher_path)

        state_dict = torch.load(pretrain_path)

        c2g = self.gene_match_idx['conv1_2g']
        c2o = self.gene_match_idx['conv1_2o']
        self.state_dict()['conv1.weight'].copy_(state_dict['base.GeneChain.conv1.weight'][c2g])
        # self.state_dict()['bn1.weight'].copy_(state_dict['base.GeneChain.bn1.weight'][c2g])
        # self.state_dict()['bn1.bias'].copy_(state_dict['base.GeneChain.bn1.bias'][c2g])
        for k,v in c2o.items():
            self.state_dict()['bn1.weight'][k].copy_(state_dict['base.bn1.weight'][v].mean(dim=0))
            self.state_dict()['bn1.bias'][k].copy_(state_dict['base.bn1.bias'][v].mean(dim=0))

        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for layer_idx,layer in enumerate(layers):
            for block_idx,block in enumerate(layer):
                prefix = 'base.layer'+str(layer_idx+1)+'.'+str(block_idx)+'.'
                gene_prefix = 'base.GeneChain.layer'+str(layer_idx+1)+'.'+str(block_idx)+'.'
                # 1
                c2g = block.gene_match_idx['conv1_2g']
                c2o = block.gene_match_idx['conv1_2o']
                if block_idx == 0:
                    if layer_idx == 0:
                        lc2o = self.gene_match_idx['conv1_2o']
                    else:
                        lc2o = self.__getattr__('layer'+str(layer_idx))[0].gene_match_idx['downsample_2o']
                else:
                    lc2o = self.__getattr__('layer'+str(layer_idx+1))[0].gene_match_idx['downsample_2o']
                assert block.state_dict()['conv1.weight'].shape[1] == len(lc2o)
                for k,v in lc2o.items():
                    block.state_dict()['conv1.weight'][:,k].copy_(state_dict[gene_prefix+'conv1.weight'][c2g][:,v].sum(dim=1))
                for k,v in c2o.items():
                    block.state_dict()['bn1.weight'][k].copy_(state_dict[prefix+'bn1.weight'][v].mean(dim=0))
                    block.state_dict()['bn1.bias'][k].copy_(state_dict[prefix+'bn1.bias'][v].mean(dim=0))
                # block.state_dict()['bn1.weight'].copy_(state_dict[gene_prefix+'bn1.weight'][c2g])
                # block.state_dict()['bn1.bias'].copy_(state_dict[gene_prefix+'bn1.bias'][c2g])
                # 2
                c2g = block.gene_match_idx['conv2_2g']
                c2o = block.gene_match_idx['conv2_2o']
                lc2o = block.gene_match_idx['conv1_2o']
                assert block.state_dict()['conv2.weight'].shape[1] == len(lc2o)
                for k,v in lc2o.items():
                    block.state_dict()['conv2.weight'][:,k].copy_(state_dict[gene_prefix+'conv2.weight'][c2g][:,v].sum(dim=1))
                for k,v in c2o.items():
                    block.state_dict()['bn2.weight'][k].copy_(state_dict[prefix+'bn2.weight'][v].mean(dim=0))
                    block.state_dict()['bn2.bias'][k].copy_(state_dict[prefix+'bn2.bias'][v].mean(dim=0))
                # block.state_dict()['bn2.weight'].copy_(state_dict[gene_prefix+'bn2.weight'][c2g])
                # block.state_dict()['bn2.bias'].copy_(state_dict[gene_prefix+'bn2.bias'][c2g])
                # 3
                c2g = block.gene_match_idx['conv3_2g']
                c2o = block.gene_match_idx['conv3_2o']
                lc2o = block.gene_match_idx['conv2_2o']
                assert block.state_dict()['conv3.weight'].shape[1] == len(lc2o)
                for k,v in lc2o.items():
                    block.state_dict()['conv3.weight'][:,k].copy_(state_dict[gene_prefix+'conv3.weight'][c2g][:,v].sum(dim=1))
                for k,v in c2o.items():
                    block.state_dict()['bn3.weight'][k].copy_(state_dict[prefix+'bn3.weight'][v].mean(dim=0))
                    block.state_dict()['bn3.bias'][k].copy_(state_dict[prefix+'bn3.bias'][v].mean(dim=0))
                # block.state_dict()['bn3.weight'].copy_(state_dict[gene_prefix+'bn3.weight'][c2g])
                # block.state_dict()['bn3.bias'].copy_(state_dict[gene_prefix+'bn3.bias'][c2g])
                # downsample
                if block.downsample is not None:
                    c2g = block.gene_match_idx['downsample_2g']
                    c2o = block.gene_match_idx['downsample_2o']
                    if layer_idx == 0:
                        lc2o = self.gene_match_idx['conv1_2o']
                    else:
                        lc2o = self.__getattr__('layer'+str(layer_idx))[0].gene_match_idx['downsample_2o']
                    assert block.state_dict()['downsample.0.weight'].shape[1] == len(lc2o)
                    for k,v in lc2o.items():
                        block.state_dict()['downsample.0.weight'][:,k].copy_(state_dict[gene_prefix+'downsample.0.weight'][c2g][:,v].sum(dim=1))
                    for k,v in c2o.items():
                        block.state_dict()['downsample.1.weight'][k].copy_(state_dict[prefix+'downsample.1.weight'][v].mean(dim=0))
                        block.state_dict()['downsample.1.bias'][k].copy_(state_dict[prefix+'downsample.1.bias'][v].mean(dim=0))
                    # block.state_dict()['downsample.1.weight'].copy_(state_dict[gene_prefix+'downsample.1.weight'][c2g])
                    # block.state_dict()['downsample.1.bias'].copy_(state_dict[gene_prefix+'downsample.1.bias'][c2g])

    def Init_norm(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                sample_filter = torch.normal(0, math.sqrt(2./n), size=m.weight[0].size())
                sample_norm = torch.norm(sample_filter)
                sample_ratio = (sample_norm / m.weight.view(m.weight.shape[0],-1).norm(dim=1)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                m.weight.data.copy_(m.weight.data *  sample_ratio)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # add missed relu
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

