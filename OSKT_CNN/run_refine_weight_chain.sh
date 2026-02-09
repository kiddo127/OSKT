


# Market1501 (weight_chain_in_planes: 8)
python3 train.py --TRAIN_DATASET Market1501 --VAL_DATASETS Market1501 \
    --DEVICE_ID '2' \
    --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 20 --LOG_PERIOD 100 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 \
    --refining_weight_chain True \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/original/LUP_pretrain/Market1501/resnet50_200.pth \
    --in_planes 64 --multipliers 2 4 8 --weight_chain_in_planes 8 \
    --MAX_EPOCHS 100 --STEPS 50 70 \

# MSMT17_v1 (weight_chain_in_planes: 8)
python3 train.py --TRAIN_DATASET MSMT17_v1 --VAL_DATASETS MSMT17_v1 \
    --DEVICE_ID '2' \
    --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 20 --LOG_PERIOD 200 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 \
    --refining_weight_chain True \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/original/LUP_pretrain/MSMT17_v1/resnet50_200.pth \
    --in_planes 64 --multipliers 2 4 8 --weight_chain_in_planes 8 \
    --MAX_EPOCHS 100 --STEPS 50 70 \

# Market1501 (weight_chain_in_planes: 16)
python3 train.py --TRAIN_DATASET Market1501 --VAL_DATASETS Market1501 \
    --DEVICE_ID '2' \
    --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 20 --LOG_PERIOD 100 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 \
    --refining_weight_chain True \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/original/LUP_pretrain/Market1501/resnet50_200.pth \
    --in_planes 64 --multipliers 2 4 8 --weight_chain_in_planes 16 \
    --MAX_EPOCHS 100 --STEPS 50 70 \

# MSMT17_v1 (weight_chain_in_planes: 16)
python3 train.py --TRAIN_DATASET MSMT17_v1 --VAL_DATASETS MSMT17_v1 \
    --DEVICE_ID '2' \
    --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 20 --LOG_PERIOD 200 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 \
    --refining_weight_chain True \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/original/LUP_pretrain/MSMT17_v1/resnet50_200.pth \
    --in_planes 64 --multipliers 2 4 8 --weight_chain_in_planes 16 \
    --MAX_EPOCHS 100 --STEPS 50 70 \

# Market1501 (weight_chain_in_planes: 32)
python3 train.py --TRAIN_DATASET Market1501 --VAL_DATASETS Market1501 \
    --DEVICE_ID '2' \
    --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 20 --LOG_PERIOD 100 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 \
    --refining_weight_chain True \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/original/LUP_pretrain/Market1501/resnet50_200.pth \
    --in_planes 64 --multipliers 2 4 8 --weight_chain_in_planes 32 \
    --MAX_EPOCHS 100 --STEPS 50 70 \

# MSMT17_v1 (weight_chain_in_planes: 32)
python3 train.py --TRAIN_DATASET MSMT17_v1 --VAL_DATASETS MSMT17_v1 \
    --DEVICE_ID '2' \
    --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 20 --LOG_PERIOD 200 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 \
    --refining_weight_chain True \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/original/LUP_pretrain/MSMT17_v1/resnet50_200.pth \
    --in_planes 64 --multipliers 2 4 8 --weight_chain_in_planes 32 \
    --MAX_EPOCHS 100 --STEPS 50 70 \


# refined from last level
# python3 train.py --TRAIN_DATASET Market1501 --VAL_DATASETS Market1501 \
#     --DEVICE_ID '2' \
#     --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 20 --LOG_PERIOD 100 \
#     --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 \
#     --refining_weight_chain True --weight_chain_in_planes 4 \
#     --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/student_model/sft/8_248/Market1501/resnet50_100.pth \
#     --in_planes 8 --multipliers 2 4 8 \
#     --MAX_EPOCHS 100 --STEPS 50 70 \

