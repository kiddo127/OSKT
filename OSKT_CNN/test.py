import os
import torch
from torch.backends import cudnn
import parser
from config import Config
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger



def collapse_model(model, pretrained_model_statedict):
    for i in model.state_dict():
        if "classifier" in i:
            continue
        if 'bn' in i or 'downsample.1' in i or 'bottleneck' in i:
            if 'num_batches_tracked' in i:
                continue
            else:
                planes = model.state_dict()[i].shape[0]
                model.state_dict()[i].copy_(pretrained_model_statedict[i][:planes])
        elif 'conv' in i or 'downsample.0' in i:
            planes = model.state_dict()[i].shape[0]
            inplanes = model.state_dict()[i].shape[1]
            weight_sum = 0
            for p in range(pretrained_model_statedict[i].shape[1]//inplanes):
                weight_sum += pretrained_model_statedict[i][:planes,p*inplanes:(p+1)*inplanes]
            model.state_dict()[i].copy_(weight_sum)
        else:
            print(i)

    return model

if __name__ == "__main__":
    
    args = parser.parse_args()
    cfg = Config()
    for arg in vars(args):
        setattr(cfg, arg, getattr(args, arg))
    
    if not os.path.exists(cfg.LOG_DIR):
        os.mkdir(cfg.LOG_DIR)
    logger = setup_logger('{}.test'.format(cfg.PROJECT_NAME), cfg.LOG_DIR)
    logger.info("Config Setting:\n{}".format([arg+': '+str(getattr(cfg, arg)) for arg in vars(cfg)])) #记录config设置
    
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.DEVICE_ID
    cudnn.benchmark = True

    _, num_classes, val_loaders = make_dataloader(cfg)
    model = make_model(cfg, num_classes)

    if cfg.training_student_model:
        model = collapse_model(model, torch.load(cfg.TEST_WEIGHT)) # 构建student_model用于测试性能
    else:
        model.load_param(cfg.TEST_WEIGHT)

    for (val_name, val_loader, num_query) in val_loaders:
        do_inference(cfg,
                    model,
                    val_name,
                    val_loader,
                    num_query)
