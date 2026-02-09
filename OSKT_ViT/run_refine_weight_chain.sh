#Single GPU


# region vit_base

# Market1501
python train.py --config_file configs/vit_base.yml \
    refining_weight_chain True weight_chain_dim 264 \
    MODEL.DEVICE_ID '("3")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_base/original/Market1501/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('Market1501')" DATASETS.VAL_NAMES "('Market1501')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 20 SOLVER.CHECKPOINT_PERIOD 60 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \

# MSMT17_v1
python train.py --config_file configs/vit_base.yml \
    refining_weight_chain True weight_chain_dim 264 \
    MODEL.DEVICE_ID '("3")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_base/original/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('MSMT17_v1')" DATASETS.VAL_NAMES "('MSMT17_v1')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 60 SOLVER.CHECKPOINT_PERIOD 60 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \

# endregion


# region vit_small

# Market1501
python train.py --config_file configs/vit_small.yml \
    refining_weight_chain True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/original/Market1501/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('Market1501')" DATASETS.VAL_NAMES "('Market1501')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 20 SOLVER.CHECKPOINT_PERIOD 60 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \

# MSMT17_v1
python train.py --config_file configs/vit_small.yml \
    refining_weight_chain True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("0")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/original/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('MSMT17_v1')" DATASETS.VAL_NAMES "('MSMT17_v1')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 60 SOLVER.CHECKPOINT_PERIOD 60 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \

# endregion



