



# region few_shot_seed 1 --few_shot_ratio 0.1
python3 train.py --TRAIN_DATASET Market1501 --VAL_DATASETS Market1501 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/Market1501 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 50 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 1 --few_shot_ratio 0.1 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 50 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 1 --few_shot_ratio 0.1 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 20 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/Market1501/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 1 --few_shot_ratio 0.1 \
#endregion


# region few_shot_seed 1 --few_shot_ratio 0.3
python3 train.py --TRAIN_DATASET Market1501 --VAL_DATASETS Market1501 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/Market1501 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 10 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 1 --few_shot_ratio 0.3 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 50 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 1 --few_shot_ratio 0.3 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 20 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/Market1501/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 1 --few_shot_ratio 0.3 \
#endregion


# region few_shot_seed 1 --few_shot_ratio 0.5
python3 train.py --TRAIN_DATASET Market1501 --VAL_DATASETS Market1501 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/Market1501 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 10 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 1 --few_shot_ratio 0.5 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 50 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 1 --few_shot_ratio 0.5 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 20 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/Market1501/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 1 --few_shot_ratio 0.5 \
#endregion




# region few_shot_seed 2 --few_shot_ratio 0.1
python3 train.py --TRAIN_DATASET Market1501 --VAL_DATASETS Market1501 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/Market1501 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 10 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 2 --few_shot_ratio 0.1 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 50 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 2 --few_shot_ratio 0.1 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 20 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/Market1501/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 2 --few_shot_ratio 0.1 \
#endregion


# region few_shot_seed 2 --few_shot_ratio 0.3
python3 train.py --TRAIN_DATASET Market1501 --VAL_DATASETS Market1501 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/Market1501 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 10 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 2 --few_shot_ratio 0.3 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 50 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 2 --few_shot_ratio 0.3 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 20 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/Market1501/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 2 --few_shot_ratio 0.3 \
#endregion


# region few_shot_seed 2 --few_shot_ratio 0.5
python3 train.py --TRAIN_DATASET Market1501 --VAL_DATASETS Market1501 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/Market1501 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 10 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 2 --few_shot_ratio 0.5 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 50 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 2 --few_shot_ratio 0.5 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 20 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/Market1501/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 2 --few_shot_ratio 0.5 \
#endregion


# region few_shot_seed 3 --few_shot_ratio 0.1
python3 train.py --TRAIN_DATASET Market1501 --VAL_DATASETS Market1501 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/Market1501 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 10 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 3 --few_shot_ratio 0.1 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 50 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 3 --few_shot_ratio 0.1 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 20 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/Market1501/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 3 --few_shot_ratio 0.1 \
#endregion


# region few_shot_seed 3 --few_shot_ratio 0.3
python3 train.py --TRAIN_DATASET Market1501 --VAL_DATASETS Market1501 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/Market1501 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 10 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 3 --few_shot_ratio 0.3 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 50 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 3 --few_shot_ratio 0.3 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 20 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/Market1501/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 3 --few_shot_ratio 0.3 \
#endregion


# region few_shot_seed 3 --few_shot_ratio 0.5
python3 train.py --TRAIN_DATASET Market1501 --VAL_DATASETS Market1501 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/Market1501 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 10 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 3 --few_shot_ratio 0.5 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 50 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 3 --few_shot_ratio 0.5 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 20 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/Market1501/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 3 --few_shot_ratio 0.5 \
#endregion





# region few_shot_seed 4 --few_shot_ratio 0.1
python3 train.py --TRAIN_DATASET Market1501 --VAL_DATASETS Market1501 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/Market1501 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 10 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 4 --few_shot_ratio 0.1 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 50 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 4 --few_shot_ratio 0.1 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 20 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/Market1501/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 4 --few_shot_ratio 0.1 \
#endregion


# region few_shot_seed 4 --few_shot_ratio 0.3
python3 train.py --TRAIN_DATASET Market1501 --VAL_DATASETS Market1501 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/Market1501 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 10 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 4 --few_shot_ratio 0.3 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 50 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 4 --few_shot_ratio 0.3 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 20 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/Market1501/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 4 --few_shot_ratio 0.3 \
#endregion


# region few_shot_seed 4 --few_shot_ratio 0.5
python3 train.py --TRAIN_DATASET Market1501 --VAL_DATASETS Market1501 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/Market1501 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 10 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 4 --few_shot_ratio 0.5 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 50 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 4 --few_shot_ratio 0.5 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 20 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/Market1501/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 4 --few_shot_ratio 0.5 \
#endregion


# region few_shot_seed 5 --few_shot_ratio 0.1
python3 train.py --TRAIN_DATASET Market1501 --VAL_DATASETS Market1501 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/Market1501 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 10 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 5 --few_shot_ratio 0.1 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 50 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 5 --few_shot_ratio 0.1 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 20 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/Market1501/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 5 --few_shot_ratio 0.1 \
#endregion


# region few_shot_seed 5 --few_shot_ratio 0.3
python3 train.py --TRAIN_DATASET Market1501 --VAL_DATASETS Market1501 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/Market1501 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 10 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 5 --few_shot_ratio 0.3 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 50 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 5 --few_shot_ratio 0.3 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 20 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/Market1501/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 5 --few_shot_ratio 0.3 \
#endregion


# region few_shot_seed 5 --few_shot_ratio 0.5
python3 train.py --TRAIN_DATASET Market1501 --VAL_DATASETS Market1501 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/Market1501 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 10 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 5 --few_shot_ratio 0.5 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 50 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/MSMT17_v1/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 5 --few_shot_ratio 0.5 \

python3 train.py --TRAIN_DATASET CUHK03 --VAL_DATASETS CUHK03 \
    --DEVICE_ID '3' --OUTPUT_DIR ./output/student_model/da/CUHK03 \
    --MAX_EPOCHS 100 --CHECKPOINT_PERIOD 100 --EVAL_PERIOD 20 --STEPS 40 70 90 \
    --BATCH_SIZE 64 --NUM_IMG_PER_ID 4 --LOG_PERIOD 100 \
    --PRETRAIN_CHOICE reid --PRETRAIN_PATH ./output/weight_chain/8/Market1501/resnet50_100.pth \
    --in_planes 8 --multipliers 2 4 8 \
    --training_student_model True \
    --few_shot_seed 5 --few_shot_ratio 0.5 \
#endregion

