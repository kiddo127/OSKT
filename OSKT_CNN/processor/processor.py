import logging
import numpy as np
import os
import time
import torch
import torch.nn as nn
import random
import json

from utils.meter import AverageMeter
from utils.metrics import R1_mAP
import copy
from solver import WarmupMultiStepLR




def save_json(file_path, data):
    assert file_path.split('.')[-1] == 'json'
    with open(file_path,'w') as file:
        json.dump(data,file)

def load_json(file_path):
    assert file_path.split('.')[-1] == 'json'
    if os.path.exists(file_path):
        with open(file_path,'r') as file:
            data = json.load(file)
    else:
        data = {}
    return data


def do_train(cfg,
             model,
             center_criterion,
             gene_center_criterion,
             train_loader,
             val_loaders,
             optimizer,
             optimizer_center,
             gene_optimizer_center,
             scheduler,
             loss_fn,
             gene_loss_func):
    log_period = cfg.LOG_PERIOD
    checkpoint_period = cfg.CHECKPOINT_PERIOD
    eval_period = cfg.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.MAX_EPOCHS

    logger = logging.getLogger('{}.train'.format(cfg.PROJECT_NAME))
    logger.info('start training')

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    loss_meter = AverageMeter()
    genenet_loss_meter = AverageMeter()
    refining_loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluators = []
    for (val_name, val_loader, num_query) in val_loaders:
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.FEAT_NORM, reranking=cfg.RERANKING)
        evaluators.append(evaluator)

    trained_epochs = []
    mAPs = []
    R1s = []

    num_iter = 0
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        genenet_loss_meter.reset()
        refining_loss_meter.reset()
        acc_meter.reset()
        scheduler.step()
        model.train()

        for n_iter, (img, pid) in enumerate(train_loader):
            num_iter += 1
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            if cfg.refining_weight_chain:
                gene_optimizer_center.zero_grad()
            img = img.to(device)
            target = pid.to(device)

            if cfg.refining_weight_chain:
                score, feat, GeneNet_score, GeneNet_feat = model(img, label=target)
            else:
                score, feat = model(img, label=target)
            loss = loss_fn(score, feat, target)
            loss_meter.update(loss.item(), img.shape[0])

            if cfg.refining_weight_chain:
                genenet_reid_loss = gene_loss_func(GeneNet_score, GeneNet_feat, target)
                genenet_loss_meter.update(genenet_reid_loss.item(), img.shape[0])
                loss += genenet_reid_loss
                refining_loss = model.base.get_refining_loss()
                refining_loss_meter.update(refining_loss.item(), 1)
                loss += refining_loss

            loss.backward()
            optimizer.step()
            if 'center' in cfg.LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.CENTER_LOSS_WEIGHT)
                optimizer_center.step()
                if cfg.refining_weight_chain:
                    for param in gene_center_criterion.parameters():
                        param.grad.data *= (1. / cfg.CENTER_LOSS_WEIGHT)
                    gene_optimizer_center.step()

            acc = (score.max(1)[1] == target).float().mean()
            acc_meter.update(acc, 1)


            if (n_iter + 1) % log_period == 0:
                out_string = "Epoch[{}] Iteration[{}/{}] "
                out_value = [epoch, (n_iter + 1), len(train_loader), loss_meter.avg]
                out_string += "Loss: {:.3f}, "
                if cfg.refining_weight_chain:
                    out_string += "GeneNet Loss: {:.3f}, "
                    out_value.append(genenet_loss_meter.avg)
                    out_string += "Refining Loss: {:.3f}, "
                    out_value.append(refining_loss_meter.avg)
                out_string += "Acc: {:.3f}, Base Lr: {:.2e}"
                out_value += [acc_meter.avg, scheduler.get_lr()[0]]
                logger.info(out_string.format(*out_value))

                loss_meter.reset()
                genenet_loss_meter.reset()
                refining_loss_meter.reset()
                acc_meter.reset()


        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL_NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            model.eval()
            for evaluator in evaluators:
                evaluator.reset()
            for i, (val_name, val_loader, num_query) in enumerate(val_loaders):
                for n_iter, (img, vid, camid, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        feat = model(img)
                        evaluators[i].update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluators[i].compute()
                logger.info(val_name+" Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))


            trained_epochs.append(epoch)
            mAPs.append(mAP)
            R1s.append(cmc[0])



def do_inference(cfg,
                 model,
                 val_name,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger('{}.test'.format(cfg.PROJECT_NAME))
    logger.info("Enter inferencing")
    evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.FEAT_NORM, \
                       method=cfg.TEST_METHOD, reranking=cfg.RERANKING)
    evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    for n_iter, (img, pid, camid, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)

            if cfg.FLIP_FEATS == 'on':
                feat = torch.FloatTensor(img.size(0), 2048).zero_().cuda()
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                    f = model(img)
                    feat = feat + f
            else:
                feat = model(img)

            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, distmat, pids, camids, qfeats, gfeats = evaluator.compute()
    logger.info(val_name+"Validation Results")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

