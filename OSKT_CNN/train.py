import os
import torch
from torch.backends import cudnn
import parser
import math
from config import Config
from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer, WarmupMultiStepLR, cosine_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from loss import make_loss
from processor import do_train, do_inference
from loss.build_criterion import ReIDLoss




def find_all_indices(lst, label):
    return [index for index, value in enumerate(lst) if value == label]

def init_classifier(model, train_loader, num_classes, feat_dim):
    print(model.classifier.weight[0][0])
    class_features_sum = torch.zeros(num_classes, feat_dim)
    class_samples_count = torch.zeros(num_classes)
    model.eval()
    with torch.no_grad():
        for inputs, labels in train_loader:
            outputs = model(inputs,setting_classifier=True)
            features = outputs.detach()
            for i, label in enumerate(labels):
                class_features_sum[label] += features[i]
                class_samples_count[label] += 1
    class_center_features = class_features_sum / class_samples_count.unsqueeze(1)

    mean = torch.mean(class_center_features, dim=0)
    var = torch.var(class_center_features, dim=0)
    normalized_class_center_features = (class_center_features - mean) / torch.sqrt(var + 1e-10) * 0.1

    with torch.no_grad():
        model.classifier.weight.copy_(normalized_class_center_features)
    print(model.classifier.weight[0][0])


if __name__ == '__main__':

    args = parser.parse_args()
    cfg = Config()
    for arg in vars(args):
        setattr(cfg, arg, getattr(args, arg))

    if cfg.training_student_model:
        model_shape = str(cfg.in_planes)+'_'+str(cfg.multipliers[0])+str(cfg.multipliers[1])+str(cfg.multipliers[2])
        if cfg.TRAIN_DATASET in cfg.PRETRAIN_PATH:
            cfg.OUTPUT_DIR = './output/student_model/sft/'+model_shape+'/'+cfg.TRAIN_DATASET
        else:
            source = cfg.PRETRAIN_PATH.split('/')[-2]
            cfg.OUTPUT_DIR = './output/student_model/da/'+model_shape+'/'+source+'_'+cfg.TRAIN_DATASET

    elif cfg.refining_weight_chain:
        if cfg.weight_chain_in_planes:
            cfg.OUTPUT_DIR = './output/weight_chain/'+str(cfg.weight_chain_in_planes)
        if cfg.in_planes < 64:
            cfg.OUTPUT_DIR = cfg.OUTPUT_DIR+'_from_'+str(cfg.in_planes)
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR+'/'+cfg.TRAIN_DATASET

    # print(cfg.OUTPUT_DIR)
    # print(cfg.PRETRAIN_PATH)
    # exit()

    if not os.path.exists(cfg.OUTPUT_DIR):
        os.mkdir(cfg.OUTPUT_DIR)
    logger = setup_logger('{}'.format(cfg.PROJECT_NAME), cfg.OUTPUT_DIR) # 输出到cfg.OUTPUT_DIR下
    logger.info("Config Setting:\n{}".format([arg+': '+str(getattr(cfg, arg)) for arg in vars(cfg)])) #记录config设置
    
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.DEVICE_ID # GPU IDs, i.e. "0,1,2" for multiple GPUs
    cudnn.benchmark = True # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.

    train_loader, num_classes, val_loaders = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes)

    if cfg.PRETRAIN_CHOICE == 'imagenet':
        print("Loading imagenet pretrained model......")
        model.base.load_param(cfg.PRETRAIN_PATH)
    elif cfg.PRETRAIN_CHOICE == 'reid':
        print("Loading reid pretrained model......")
        model.load_param(cfg.PRETRAIN_PATH)

    if cfg.refining_weight_chain:
        gene_matcher_path = os.path.join(cfg.OUTPUT_DIR,"gene_matcher.json")
        if not os.path.exists(gene_matcher_path):
            model.base.FilterClustering()
            model.base.ClusterCenter_GeneInit()
            model.base.save_gene_matcher(save_path=gene_matcher_path)
        else:
            model.base.load_gene_matcher(load_path=gene_matcher_path)
            model.base.ClusterCenter_GeneInit()
        model.base.GeneChain.trans_conv_setting()
        logger.info("Filter clustering for weight chain initialization.")
    elif cfg.training_student_model:
        gene_matcher_path = os.path.join(os.path.split(cfg.PRETRAIN_PATH)[0],"gene_matcher.json")
        model.base.Init_via_Gene(cfg.PRETRAIN_PATH, gene_matcher_path)


    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes, feat_dim=model.in_planes)
    gene_center_criterion, gene_loss_func, gene_optimizer_center = None, None, None
    if cfg.refining_weight_chain:
        gene_loss_func, gene_center_criterion = make_loss(cfg, num_classes=num_classes, feat_dim=model.weight_chain_in_planes)
        gene_optimizer_center = make_optimizer(cfg, model, gene_center_criterion, center_only=True)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    scheduler = WarmupMultiStepLR(optimizer, cfg.STEPS, cfg.GAMMA,
                                  cfg.WARMUP_FACTOR,
                                  cfg.WARMUP_EPOCHS, cfg.WARMUP_METHOD)



    do_train(
        cfg,
        model,
        center_criterion,
        gene_center_criterion,
        train_loader,
        val_loaders,
        optimizer,
        optimizer_center,
        gene_optimizer_center,
        scheduler,
        loss_func,
        gene_loss_func,
    )

    # do_inference(
    #     cfg,
    #     model,
    #     val_loader,
    #     num_query
    # )