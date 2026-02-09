# Market1501
python3 train.py --TRAIN_DATASET Market1501 --VAL_DATASETS Market1501 \
    --DEVICE_ID '1' --OUTPUT_DIR ./output/original/LUP_pretrain/Market1501 \
    --MAX_EPOCHS 200 --CHECKPOINT_PERIOD 20 --EVAL_PERIOD 20 --LOG_PERIOD 100 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 \
    --STEPS 40 70 130 \
    --PRETRAIN_CHOICE imagenet --PRETRAIN_PATH ./output/lupws_r50.pth \
    # --PRETRAIN_CHOICE imagenet --PRETRAIN_PATH ./output/resnet50-19c8e357.pth \

# MSMT17_v1
python3 train.py --TRAIN_DATASET MSMT17_v1 --VAL_DATASETS MSMT17_v1 \
    --DEVICE_ID '2' --OUTPUT_DIR ./output/original/LUP_pretrain/MSMT17_v1 \
    --MAX_EPOCHS 200 --CHECKPOINT_PERIOD 20 --EVAL_PERIOD 20 --LOG_PERIOD 200 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 \
    --STEPS 40 70 130 \
    --PRETRAIN_CHOICE imagenet --PRETRAIN_PATH ./output/lupws_r50.pth \
    # --PRETRAIN_CHOICE imagenet --PRETRAIN_PATH ./output/resnet50-19c8e357.pth \

# CUHK03
python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/original/LUP_pretrain/CUHK03 \
    --MAX_EPOCHS 200 --CHECKPOINT_PERIOD 20 --EVAL_PERIOD 20 --LOG_PERIOD 50 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 \
    --STEPS 40 70 130 \
    --PRETRAIN_CHOICE imagenet --PRETRAIN_PATH ./output/lupws_r50.pth \
    # --PRETRAIN_CHOICE imagenet --PRETRAIN_PATH ./output/resnet50-19c8e357.pth \
