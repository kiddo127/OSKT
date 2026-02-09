import torch
import torch.nn as nn
from .backbones.resnet_wGene import ResNet_wGene, CoupledResNet, Bottleneck_wGene, GeneChain, CoupledBottleneck
from .backbones.resnet import ResNet, Bottleneck
from loss.arcface import ArcFace
from loss.triplet_loss import TripletLoss
import torch.nn.functional as F
import math
import os
from torch.nn import Parameter


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)



def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model

class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()

        self.base = ResNet(last_stride=cfg.LAST_STRIDE, # self.LAST_STRIDE = 1  # the stride of the last layer of resnet50
                            block=Bottleneck,
                            layers=[3, 4, 6, 3],
                            in_planes=cfg.in_planes,
                            multipliers=cfg.multipliers)
        self.in_planes = self.base.in_planes
        self.gap = nn.AdaptiveAvgPool2d(1) #nn.functional.avg_pool2d
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = self.gap(x) #(batch_size,channel_num,1,1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        feat = self.bottleneck(global_feat)
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            return feat

    def load_param(self, trained_path):
        print('Loading pretrained model from {}'.format(trained_path))
        param_dict = torch.load(trained_path)
        if 'pth.tar' in trained_path:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            if "Gene" in i or "gene" in i:
                continue
            try:
                self.state_dict()[i.replace('module.','')].copy_(param_dict[i])
            except Exception as e:
                pass



class Backbone_wGene(nn.Module):
    def __init__(self, num_classes, cfg):
        super().__init__()

        self.base = ResNet_wGene(last_stride=cfg.LAST_STRIDE, # self.LAST_STRIDE = 1  # the stride of the last layer of resnet50
                                 block=Bottleneck_wGene,
                                 layers=[3, 4, 6, 3],
                                 in_planes=cfg.in_planes,
                                 multipliers=cfg.multipliers,
                                 weight_chain_in_planes=cfg.weight_chain_in_planes)
        self.in_planes = self.base.in_planes
        self.gap = nn.AdaptiveAvgPool2d(1) # nn.functional.avg_pool2d
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        # GeneNet
        self.weight_chain_in_planes = self.base.GeneChain.weight_chain_in_planes
        self.GeneNet_bottleneck = nn.BatchNorm1d(self.weight_chain_in_planes)
        self.GeneNet_bottleneck.bias.requires_grad_(False)
        self.GeneNet_bottleneck.apply(weights_init_kaiming)
        self.GeneNet_classifier = nn.Linear(self.weight_chain_in_planes, num_classes, bias=False)
        self.GeneNet_classifier.apply(weights_init_classifier)
        # GeneNet

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        # x = self.base(x)
        x,GeneNet_out = self.base(x)

        global_feat = self.gap(x) #(batch_size,channel_num,1,1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        feat = self.bottleneck(global_feat)
        # feat = F.normalize(global_feat,dim=1)

        # GeneNet
        GeneNet_global_feat = self.gap(GeneNet_out) #(batch_size,channel_num,1,1)
        GeneNet_global_feat = GeneNet_global_feat.view(GeneNet_global_feat.shape[0], -1)  # flatten to (bs, 2048)
        GeneNet_feat = self.GeneNet_bottleneck(GeneNet_global_feat)
        if self.training:
            cls_score = self.classifier(feat)
            GeneNet_cls_score = self.GeneNet_classifier(GeneNet_feat)
            return cls_score, global_feat, GeneNet_cls_score, GeneNet_global_feat
        else:
            return feat


    def load_param(self, trained_path):
        print('Loading pretrained model from {}'.format(trained_path))
        param_dict = torch.load(trained_path)
        if 'pth.tar' in trained_path:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            try:
                self.state_dict()[i.replace('module.','')].copy_(param_dict[i])
            except Exception as e:
                pass



class CoupledBackbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super().__init__()
        self.base = CoupledResNet(
            last_stride=cfg.LAST_STRIDE, # self.LAST_STRIDE = 1  # the stride of the last layer of resnet50
            block=CoupledBottleneck,
            layers=[3, 4, 6, 3],
            in_planes=cfg.in_planes,
            multipliers=cfg.multipliers
            )

        self.in_planes = self.base.in_planes

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x, label=None, setting_classifier=False):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = self.gap(x) #(batch_size,channel_num,1,1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        feat = self.bottleneck(global_feat)
        if setting_classifier:
            return feat
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            return feat

    def load_param(self, trained_path):
        print('Loading pretrained model from {}'.format(trained_path))
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if "Gene" in i or "gene" in i:
                continue
            try:
                self.state_dict()[i.replace('module.','')].copy_(param_dict[i])
            except Exception as e:
                pass




class Backbone_Vanilla_LG(Backbone):
    def __init__(self, num_classes, cfg):
        super().__init__(num_classes, cfg)

        layers = [1,1,1,3]
        self.base = ResNet(last_stride=cfg.LAST_STRIDE, # self.LAST_STRIDE = 1  # the stride of the last layer of resnet50
                            block=Bottleneck,
                            layers=layers)

    def random_expansion(self, trained_path):
        print('Loading pretrained model from {}'.format(trained_path))
        param_dict = torch.load(trained_path)
        if 'pth.tar' in trained_path:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            if 'layer4' not in i and 'bottleneck' not in i and 'classifier' not in i:
                continue
            try:
                self.state_dict()[i.replace('module.','')].copy_(param_dict[i])
            except Exception as e:
                pass


class Backbone_SWS_LG(Backbone):
    def __init__(self, num_classes, cfg):
        super().__init__(num_classes, cfg)

        layers = [1,1,1,1]
        self.base = ResNet(last_stride=cfg.LAST_STRIDE, # self.LAST_STRIDE = 1  # the stride of the last layer of resnet50
                            block=Bottleneck,
                            layers=layers)

    def random_expansion(self, trained_path):
        print('Loading pretrained model from {}'.format(trained_path))
        param_dict = torch.load(trained_path)
        if 'pth.tar' in trained_path:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            if 'layer' in i and ('layer1.0' not in i and 'layer2.0' not in i and 'layer3.0' not in i and 'layer4.0' not in i):
                continue
            try:
                self.state_dict()[i.replace('module.','')].copy_(param_dict[i])
            except Exception as e:
                pass



def make_model(cfg, num_class):
    if cfg.refining_weight_chain:
        model = Backbone_wGene(num_class, cfg)
    elif cfg.training_student_model:
        model = CoupledBackbone(num_class, cfg)
    else:
        model = Backbone(num_class, cfg)
    return model
