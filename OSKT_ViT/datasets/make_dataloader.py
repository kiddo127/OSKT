import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from timm.data.random_erasing import RandomErasing
from .bases import ImageDataset
from .sampler import RandomIdentitySampler, RandomIdentitySampler_IdUniform
from .DatasetsLoader import TrainDatasetsBalanced, Market1501, DukeMTMCreID, CUHK03NP, MSMT17_v1, MSMT17, CUHK03
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .mm import MM

__factory = {
    'TrainDatasetsBalanced': TrainDatasetsBalanced,
    'Market1501': Market1501,
    'CUHK03NP': CUHK03NP,
    'CUHK03': CUHK03,
    'DukeMTMCreID': DukeMTMCreID,
    'MSMT17_v1': MSMT17_v1,
    'MSMT17': MSMT17,
    'mm': MM,
}

def train_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids

def val_collate_fn(batch):
    imgs, pids, camids, img_paths = zip(*batch)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, img_paths

def make_dataloader(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    # dataset = __factory[cfg.DATASETS.TRAIN_NAMES]()
    dataset = __factory[cfg.DATASETS.TRAIN_NAMES](few_shot_ratio=cfg.few_shot_ratio, few_shot_seed=cfg.few_shot_seed)

    train_set = ImageDataset(dataset.train, train_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams

    if cfg.DATALOADER.SAMPLER in ['softmax_triplet', 'img_triplet']:
        print('using img_triplet sampler')
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    elif cfg.DATALOADER.SAMPLER in ['id_triplet', 'id']:
        print('using ID sampler')
        train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler_IdUniform(dataset.train, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn, drop_last = True,
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))


    val_dataset = __factory[cfg.DATASETS.VAL_NAMES]()
    val_set = ImageDataset(val_dataset.query + val_dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    return train_loader, val_loader, len(val_dataset.query), num_classes, cam_num
