
# student Res-50-S1: --in_planes 8 --multipliers 2 4 8
python3 train.py --TRAIN_DATASET Market1501 --VAL_DATASETS Market1501 \
    --DEVICE_ID '1' \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 50 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/Market1501/resnet50_100.pth \
    --training_student_model True --in_planes 8 --multipliers 2 4 8 \

# student Res-50-S2: --in_planes 16 --multipliers 2 3 4
python3 train.py --TRAIN_DATASET Market1501 --VAL_DATASETS Market1501 \
    --DEVICE_ID '1' \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 50 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/Market1501/resnet50_100.pth \
    --training_student_model True --in_planes 16 --multipliers 2 3 4 \

# student Res-50-S3: --in_planes 16 --multipliers 2 4 8
python3 train.py --TRAIN_DATASET Market1501 --VAL_DATASETS Market1501 \
    --DEVICE_ID '1' \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 50 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/16/Market1501/resnet50_100.pth \
    --training_student_model True --in_planes 16 --multipliers 2 4 8 \

# student Res-50-S4: --in_planes 32 --multipliers 2 3 4
python3 train.py --TRAIN_DATASET Market1501 --VAL_DATASETS Market1501 \
    --DEVICE_ID '1' \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 50 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/16/Market1501/resnet50_100.pth \
    --training_student_model True --in_planes 32 --multipliers 2 3 4 \

# student Res-50-S5: --in_planes 32 --multipliers 2 4 8
python3 train.py --TRAIN_DATASET Market1501 --VAL_DATASETS Market1501 \
    --DEVICE_ID '1' \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 50 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/32/Market1501/resnet50_100.pth \
    --training_student_model True --in_planes 32 --multipliers 2 4 8 \

# student Res-50-S6: --in_planes 64 --multipliers 2 3 4
python3 train.py --TRAIN_DATASET Market1501 --VAL_DATASETS Market1501 \
    --DEVICE_ID '1' \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 50 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/32/Market1501/resnet50_100.pth \
    --training_student_model True --in_planes 64 --multipliers 2 3 4 \

