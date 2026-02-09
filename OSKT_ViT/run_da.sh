


# region vit_base (weight_chain_dim: 264)

# MS->M ViT-B-S1 (student_dim: 264)
python train.py --config_file configs/vit_base.yml \
    student_dim 264 \
    training_student_model True weight_chain_dim 264 \
    MODEL.DEVICE_ID '("2")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_base/weight_chain/264/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('Market1501')" DATASETS.VAL_NAMES "('Market1501')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 30 SOLVER.CHECKPOINT_PERIOD 120 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \

# MS->M ViT-B-S2 (student_dim: 336)
python train.py --config_file configs/vit_base.yml \
    student_dim 336 \
    training_student_model True weight_chain_dim 264 \
    MODEL.DEVICE_ID '("2")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_base/weight_chain/264/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('Market1501')" DATASETS.VAL_NAMES "('Market1501')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 10 SOLVER.CHECKPOINT_PERIOD 120 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \

# MS->C ViT-B-S1 (student_dim: 264)
python train.py --config_file configs/vit_base.yml \
    student_dim 264 \
    training_student_model True weight_chain_dim 264 \
    MODEL.DEVICE_ID '("2")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_base/weight_chain/264/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 30 SOLVER.CHECKPOINT_PERIOD 120 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \

# MS->C ViT-B-S2 (student_dim: 336)
python train.py --config_file configs/vit_base.yml \
    student_dim 336 \
    training_student_model True weight_chain_dim 264 \
    MODEL.DEVICE_ID '("2")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_base/weight_chain/264/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 30 SOLVER.CHECKPOINT_PERIOD 120 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \

# M->C ViT-B-S1 (student_dim: 264)
python train.py --config_file configs/vit_base.yml \
    student_dim 264 \
    training_student_model True weight_chain_dim 264 \
    MODEL.DEVICE_ID '("2")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_base/weight_chain/264/Market1501/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 30 SOLVER.CHECKPOINT_PERIOD 120 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \

# M->C ViT-B-S2 (student_dim: 336)
python train.py --config_file configs/vit_base.yml \
    student_dim 336 \
    training_student_model True weight_chain_dim 264 \
    MODEL.DEVICE_ID '("2")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_base/weight_chain/264/Market1501/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 30 SOLVER.CHECKPOINT_PERIOD 120 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \

# endregion


# region vit_small (weight_chain_dim: 120)

# MS->M ViT-S-S1 (student_dim: 120)
python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("2")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('Market1501')" DATASETS.VAL_NAMES "('Market1501')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 30 SOLVER.CHECKPOINT_PERIOD 60 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \

# MS->M ViT-S-S2 (student_dim: 168)
python train.py --config_file configs/vit_small.yml \
    student_dim 168 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("2")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('Market1501')" DATASETS.VAL_NAMES "('Market1501')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 30 SOLVER.CHECKPOINT_PERIOD 60 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \

# MS->C ViT-S-S1 (student_dim: 120)
python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("2")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 30 SOLVER.CHECKPOINT_PERIOD 60 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \

# MS->C ViT-S-S2 (student_dim: 168)
python train.py --config_file configs/vit_small.yml \
    student_dim 168 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("2")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 30 SOLVER.CHECKPOINT_PERIOD 60 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \

# M->C ViT-S-S1 (student_dim: 120)
python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("2")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/Market1501/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 30 SOLVER.CHECKPOINT_PERIOD 60 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \

# M->C ViT-S-S2 (student_dim: 168)
python train.py --config_file configs/vit_small.yml \
    student_dim 168 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("2")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/Market1501/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 30 SOLVER.CHECKPOINT_PERIOD 60 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \

# endregion


