

# region few_shot_seed 1 few_shot_ratio 0.1
python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('Market1501')" DATASETS.VAL_NAMES "('Market1501')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 1 few_shot_ratio 0.1 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 1 few_shot_ratio 0.1 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/Market1501/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 1 few_shot_ratio 0.1 \
# endregion




# region few_shot_seed 1 few_shot_ratio 0.3
python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('Market1501')" DATASETS.VAL_NAMES "('Market1501')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 1 few_shot_ratio 0.3 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 1 few_shot_ratio 0.3 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/Market1501/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 1 few_shot_ratio 0.3 \
# endregion





# region few_shot_seed 1 few_shot_ratio 0.5
python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('Market1501')" DATASETS.VAL_NAMES "('Market1501')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 1 few_shot_ratio 0.5 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 1 few_shot_ratio 0.5 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/Market1501/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 1 few_shot_ratio 0.5 \
# endregion





# region few_shot_seed 2 few_shot_ratio 0.1
python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('Market1501')" DATASETS.VAL_NAMES "('Market1501')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 2 few_shot_ratio 0.1 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 2 few_shot_ratio 0.1 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/Market1501/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 2 few_shot_ratio 0.1 \
# endregion




# region few_shot_seed 2 few_shot_ratio 0.3
python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('Market1501')" DATASETS.VAL_NAMES "('Market1501')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 2 few_shot_ratio 0.3 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 2 few_shot_ratio 0.3 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/Market1501/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 2 few_shot_ratio 0.3 \
# endregion





# region few_shot_seed 2 few_shot_ratio 0.5
python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('Market1501')" DATASETS.VAL_NAMES "('Market1501')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 2 few_shot_ratio 0.5 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 2 few_shot_ratio 0.5 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/Market1501/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 2 few_shot_ratio 0.5 \
# endregion





# region few_shot_seed 3 few_shot_ratio 0.1
python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('Market1501')" DATASETS.VAL_NAMES "('Market1501')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 3 few_shot_ratio 0.1 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 3 few_shot_ratio 0.1 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/Market1501/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 3 few_shot_ratio 0.1 \
# endregion




# region few_shot_seed 3 few_shot_ratio 0.3
python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('Market1501')" DATASETS.VAL_NAMES "('Market1501')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 3 few_shot_ratio 0.3 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 3 few_shot_ratio 0.3 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/Market1501/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 3 few_shot_ratio 0.3 \
# endregion





# region few_shot_seed 3 few_shot_ratio 0.5
python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('Market1501')" DATASETS.VAL_NAMES "('Market1501')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 3 few_shot_ratio 0.5 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 3 few_shot_ratio 0.5 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/Market1501/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 3 few_shot_ratio 0.5 \
# endregion




# region few_shot_seed 4 few_shot_ratio 0.1
python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('Market1501')" DATASETS.VAL_NAMES "('Market1501')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 4 few_shot_ratio 0.1 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 4 few_shot_ratio 0.1 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/Market1501/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 4 few_shot_ratio 0.1 \
# endregion




# region few_shot_seed 4 few_shot_ratio 0.3
python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('Market1501')" DATASETS.VAL_NAMES "('Market1501')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 4 few_shot_ratio 0.3 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 4 few_shot_ratio 0.3 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/Market1501/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 4 few_shot_ratio 0.3 \
# endregion





# region few_shot_seed 4 few_shot_ratio 0.5
python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('Market1501')" DATASETS.VAL_NAMES "('Market1501')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 4 few_shot_ratio 0.5 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 4 few_shot_ratio 0.5 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/Market1501/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 4 few_shot_ratio 0.5 \
# endregion



# region few_shot_seed 5 few_shot_ratio 0.1
python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('Market1501')" DATASETS.VAL_NAMES "('Market1501')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 5 few_shot_ratio 0.1 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 5 few_shot_ratio 0.1 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/Market1501/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 5 few_shot_ratio 0.1 \
# endregion




# region few_shot_seed 5 few_shot_ratio 0.3
python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('Market1501')" DATASETS.VAL_NAMES "('Market1501')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 5 few_shot_ratio 0.3 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 5 few_shot_ratio 0.3 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/Market1501/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 5 few_shot_ratio 0.3 \
# endregion





# region few_shot_seed 5 few_shot_ratio 0.5
python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('Market1501')" DATASETS.VAL_NAMES "('Market1501')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 5 few_shot_ratio 0.5 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/MSMT17_v1/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 5 few_shot_ratio 0.5 \

python train.py --config_file configs/vit_small.yml \
    student_dim 120 \
    training_student_model True weight_chain_dim 120 \
    MODEL.DEVICE_ID '("1")' SOLVER.SEED 42 \
    MODEL.PRETRAIN_CHOICE reid MODEL.PRETRAIN_PATH ./output/vit_small/weight_chain/120/Market1501/transformer_120.pth \
    DATASETS.TRAIN_NAMES "('CUHK03')" DATASETS.VAL_NAMES "('CUHK03')" \
    SOLVER.MAX_EPOCHS 120 SOLVER.EVAL_PERIOD 120 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.LOG_PERIOD 100 \
    SOLVER.IMS_PER_BATCH 64 SOLVER.MinLRScale 1 \
    few_shot_seed 5 few_shot_ratio 0.5 \
# endregion


