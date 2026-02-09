from utils.logger import setup_logger
from datasets import make_dataloader
from solver import make_optimizer, WarmupMultiStepLR
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
import torch.distributed as dist

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    # set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    output_dir = cfg.OUTPUT_DIR
    if cfg.refining_weight_chain:
        weight_chain_dim = str(cfg.weight_chain_dim)
        output_dir = './output/'+args.config_file.split('/')[-1].split('.')[0]+'/weight_chain/'+weight_chain_dim+'/'+cfg.DATASETS.TRAIN_NAMES
    elif cfg.training_student_model:
        model_shape = str(cfg.student_dim)
        if cfg.DATASETS.TRAIN_NAMES in cfg.MODEL.PRETRAIN_PATH:
            output_dir = './output/'+args.config_file.split('/')[-1].split('.')[0]+'/student_model/sft/'+model_shape+'/'+cfg.DATASETS.TRAIN_NAMES
        else:
            source = cfg.MODEL.PRETRAIN_PATH.split("/")[-2]
            output_dir = './output/'+args.config_file.split('/')[-1].split('.')[0]+'/student_model/da/'+model_shape+'/'+source+"_"+cfg.DATASETS.TRAIN_NAMES
    # print(output_dir)
    # exit()
    try:
        os.makedirs(output_dir)
    except:
        pass

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(output_dir))
    #  logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            #  logger.info(config_str)

    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, val_loader, num_query, num_classes, camera_num = make_dataloader(cfg)

    if cfg.refining_weight_chain or cfg.training_student_model:
        from model.make_model_WeightChain import make_model
    else:
        from model import make_model
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num)

    if cfg.refining_weight_chain:
        gene_matcher_path = os.path.join(output_dir,"gene_matcher.json")
        if os.path.exists(gene_matcher_path):
            model.base.load_gene_matcher(gene_matcher_path)
        else:
            model.base.Pretrained_Weight_Clustering()
            model.base.save_gene_matcher(gene_matcher_path)
        model.base.Cluster_Center_Init()
        model.base.learner_init()
        model.entire_head_init()

    elif cfg.training_student_model:
        gene_matcher_path = os.path.join(os.path.dirname(cfg.MODEL.PRETRAIN_PATH),"gene_matcher.json")
        model.base.load_gene_matcher(gene_matcher_path)
        model.base.learner_init()
        model.entire_head_init()
        model.load_param(cfg.MODEL.PRETRAIN_PATH)
        model = model.get_decoded_model(cfg, num_class=num_classes, camera_num=camera_num, backbone_embed_dim=cfg.student_dim)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    if cfg.SOLVER.WARMUP_METHOD == 'cosine':
        logger.info('===========using cosine learning rate=======')
        scheduler = create_scheduler(cfg, optimizer)
    else:
        logger.info('===========using normal learning rate=======')
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                      cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_EPOCHS, cfg.SOLVER.WARMUP_METHOD)

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query, args.local_rank,
        output_dir
    )
    #  print(output_dir)
    #  print(cfg.MODEL.PRETRAIN_PATH)
