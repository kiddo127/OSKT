import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
import json




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
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank,
             output_dir):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    # ===== meters & evaluator =====
    loss_meter = AverageMeter()
    p_loss_meter = AverageMeter()
    refining_loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    p_acc_meter = AverageMeter()
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    def meters_reset():
        loss_meter.reset()
        p_loss_meter.reset()
        refining_loss_meter.reset()
        acc_meter.reset()
        p_acc_meter.reset()
        evaluator.reset()

    # ===== training settings =====
    refining_weight_chain = cfg.refining_weight_chain
    scale_factor = None
    eval_scale_factor = None

    trained_epochs = []
    mAPs = []
    R1s = []

    pcoe_iter = len(train_loader)*epochs
    num_iter = 0
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        meters_reset()

        model.train()

        for n_iter, (img, vid, target_cam) in enumerate(train_loader):
            num_iter += 1
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)

            with amp.autocast(enabled=True):
                score, feat = model(img, target, cam_label=target_cam, scale_factor=scale_factor)
                loss = loss_fn(score, feat, target, target_cam)
                loss_meter.update(loss.item(), img.shape[0])
                if refining_weight_chain:
                    # 端点模型损失
                    p_score, p_feat = model(img, target, cam_label=target_cam, scale_factor='s1')
                    p_loss = loss_fn(p_score, p_feat, target, target_cam)
                    p_loss_meter.update(p_loss.item(), img.shape[0])
                    loss += p_loss
                    # refining loss
                    refining_loss = model.base.get_refining_loss()
                    refining_loss_meter.update(refining_loss.item(), 1)
                    loss_scale = num_iter/pcoe_iter
                    loss += refining_loss*loss_scale

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            try:
                if isinstance(score, list):
                    acc = (score[0].max(1)[1] == target).float().mean()
                else:
                    acc = (score.max(1)[1] == target).float().mean()
                acc_meter.update(acc, 1)
            except:
                pass

            if refining_weight_chain:
                try:
                    if isinstance(p_score, list):
                        p_acc = (p_score[0].max(1)[1] == target).float().mean()
                    else:
                        p_acc = (p_score.max(1)[1] == target).float().mean()
                    p_acc_meter.update(p_acc, 1)
                except:
                    pass

            torch.cuda.synchronize()

            if (n_iter + 1) % log_period == 0:
                out_string = "Epoch[{}] Iter[{}/{}] Loss: {:.3f},  "
                out_value = [epoch, (n_iter + 1), len(train_loader), loss_meter.avg]
                if cfg.refining_weight_chain:
                    out_string += "PNet Loss: {:.3f}, "
                    out_value.append(p_loss_meter.avg)
                    out_string += "PNet Acc: {:.3f}, "
                    out_value.append(p_acc_meter.avg)
                    out_string += "Refining Loss: {:.3f}, "
                    out_value.append(refining_loss_meter.avg)
                    out_string += "Weight of Refining Loss: {:.3f}, "
                    out_value.append(loss_scale)
                base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                out_string += "Acc: {:.3f}, Base Lr: {:.2e}"
                out_value += [acc_meter.avg, base_lr]
                logger.info(out_string.format(*out_value))

                meters_reset()

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)

        if cfg.SOLVER.WARMUP_METHOD == 'cosine':
            scheduler.step(epoch)
        else:
            scheduler.step()

        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch * (n_iter + 1), train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(output_dir, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(output_dir, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            feat = model(img, cam_label=camids, scale_factor=eval_scale_factor)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        feat = model(img, cam_label=camids, scale_factor=eval_scale_factor)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()


            trained_epochs.append(epoch)
            mAPs.append(mAP)
            R1s.append(cmc[0])

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            feat = model(img, cam_label=camids)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


