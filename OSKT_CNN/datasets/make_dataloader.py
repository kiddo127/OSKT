from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset
from .preprocessing import RandomErasing
from .sampler import RandomIdentitySampler

from .DatasetsLoader import (
    Market1501,
    DukeMTMCreID,
    CUHK03NP,
    CUHK03,
    MSMT17,
    MSMT17_v1,
    ClonedPerson,
    RandPerson,
    PersonX,
    UnrealPerson,
    TrainDatasets,
    TrainDatasetsBalanced
)

__Datasets_Loaders = {
    "Market1501": Market1501,
    "DukeMTMCreID": DukeMTMCreID,
    "CUHK03NP": CUHK03NP,
    "CUHK03": CUHK03,
    "MSMT17": MSMT17,
    "MSMT17_v1": MSMT17_v1,
    "ClonedPerson": ClonedPerson,
    "RandPerson": RandPerson,
    "PersonX": PersonX,
    "UnrealPerson": UnrealPerson,
    "TrainDatasets": TrainDatasets,
    "TrainDatasetsBalanced": TrainDatasetsBalanced
}


def train_collate_fn(batch):
    imgs, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids


def val_collate_fn(batch):
    imgs, pids, camids, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, img_paths


def make_dataloader(cfg,only_train=False):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop(cfg.INPUT_SIZE),
        # T.RandomRotation(12, resample=Image.BICUBIC, expand=False, center=None),
        # T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0),
        #                T.RandomAffine(degrees=0, translate=None, scale=[0.8, 1.2], shear=15, \
        #                               resample=Image.BICUBIC, fillcolor=0)], p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        RandomErasing(probability=0.5, sh=0.4, mean=(0.4914, 0.4822, 0.4465))
    ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    num_workers = cfg.DATALOADER_NUM_WORKERS

    verbose = not only_train
    # train_dataset = __Datasets_Loaders[cfg.TRAIN_DATASET](verbose=verbose)
    if cfg.TRAIN_DATASET == "TrainDatasetsBalanced":
        train_dataset = __Datasets_Loaders[cfg.TRAIN_DATASET](verbose=verbose)
    else:
        train_dataset = __Datasets_Loaders[cfg.TRAIN_DATASET](verbose=verbose,
                                                          few_shot_ratio=cfg.few_shot_ratio, few_shot_seed=cfg.few_shot_seed)

    num_classes = train_dataset.num_train_pids
    train_set = ImageDataset(train_dataset.train, train_transforms)

    if cfg.SAMPLER == 'triplet':
        print('using triplet sampler')
        train_loader = DataLoader(train_set,
                                batch_size=cfg.BATCH_SIZE,
                                num_workers=num_workers,
                                sampler=RandomIdentitySampler(train_dataset.train, cfg.BATCH_SIZE, cfg.NUM_IMG_PER_ID),
                                pin_memory = True, prefetch_factor = cfg.prefetch_factor,
                                collate_fn=train_collate_fn  # customized batch sampler
                                )
    elif cfg.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(train_set,
                                batch_size=cfg.BATCH_SIZE,
                                shuffle=True,
                                num_workers=num_workers,
                                sampler=None,
                                collate_fn=train_collate_fn,  # customized batch sampler
                                drop_last=True
                                )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    if only_train:
        return train_loader, num_classes

    val_loaders = []
    if cfg.VAL_DATASETS:
        val_datasets = []
        for VAL_DATASET in cfg.VAL_DATASETS:
            val_dataset = __Datasets_Loaders[VAL_DATASET](verbose=True)
            val_datasets.append((VAL_DATASET,val_dataset))
        for (dataset_name, dataset) in val_datasets:
            val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
            val_loader = DataLoader(val_set,
                                        batch_size=cfg.TEST_IMS_PER_BATCH,
                                        shuffle=False, num_workers=num_workers,
                                        pin_memory = False, prefetch_factor = cfg.prefetch_factor,
                                        collate_fn=val_collate_fn #imgs,pids,camids,img_paths
                                        )
            val_loaders.append((dataset_name, val_loader, len(dataset.query)))

    return train_loader, num_classes, val_loaders
