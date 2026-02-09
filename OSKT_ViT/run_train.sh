#Single GPU


# region vit_base

# Market1501
python train.py --config_file configs/vit_base.yml \
    MODEL.DEVICE_ID '("2")' SOLVER.SEED 42 \
    DATASETS.TRAIN_NAMES "('Market1501')" DATASETS.VAL_NAMES "('Market1501')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 20 SOLVER.CHECKPOINT_PERIOD 60 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 \
    OUTPUT_DIR ./output/vit_base/original/Market1501 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/pass_vit_base_full.pth \

# MSMT17_v1
python train.py --config_file configs/vit_base.yml \
    MODEL.DEVICE_ID '("2")' SOLVER.SEED 42 \
    DATASETS.TRAIN_NAMES "('MSMT17_v1')" DATASETS.VAL_NAMES "('MSMT17_v1')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 20 SOLVER.CHECKPOINT_PERIOD 60 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 \
    OUTPUT_DIR ./output/vit_base/original/MSMT17_v1 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/pass_vit_base_full.pth \

# endregion


# region vit_small

# Market1501
python train.py --config_file configs/vit_small.yml \
    MODEL.DEVICE_ID '("3")' SOLVER.SEED 42 \
    DATASETS.TRAIN_NAMES "('Market1501')" DATASETS.VAL_NAMES "('Market1501')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 60 SOLVER.CHECKPOINT_PERIOD 60 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 \
    OUTPUT_DIR ./output/vit_small/original/Market1501 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/pass_vit_small_full.pth \

# MSMT17_v1
python train.py --config_file configs/vit_small.yml \
    MODEL.DEVICE_ID '("3")' SOLVER.SEED 42 \
    DATASETS.TRAIN_NAMES "('MSMT17_v1')" DATASETS.VAL_NAMES "('MSMT17_v1')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 60 SOLVER.CHECKPOINT_PERIOD 60 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 \
    OUTPUT_DIR ./output/vit_small/original/MSMT17_v1 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/pass_vit_small_full.pth \

# endregion
