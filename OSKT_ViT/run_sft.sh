#Single GPU


# region vit_base (weight_chain_dim: 264)
python train.py --config_file configs/vit_base.yml \
    student_dim 336 \
    training_student_model True weight_chain_dim 264 \
    MODEL.DEVICE_ID '("3")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_base/weight_chain/264/Market1501/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('Market1501')" DATASETS.VAL_NAMES "('Market1501')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 10 SOLVER.CHECKPOINT_PERIOD 120 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \

python train.py --config_file configs/vit_base.yml \
    student_dim 264 \
    training_student_model True weight_chain_dim 264 \
    MODEL.DEVICE_ID '("3")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_base/weight_chain/264/Market1501/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('Market1501')" DATASETS.VAL_NAMES "('Market1501')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 10 SOLVER.CHECKPOINT_PERIOD 120 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
# endregion


# region vit_small (weight_chain_dim: 120)
python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("3")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/Market1501/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('Market1501')" DATASETS.VAL_NAMES "('Market1501')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 10 SOLVER.CHECKPOINT_PERIOD 120 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \

python train.py --config_file configs/vit_small.yml \
    student_dim 168 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("3")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/Market1501/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('Market1501')" DATASETS.VAL_NAMES "('Market1501')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 20 SOLVER.CHECKPOINT_PERIOD 120 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
# endregion


