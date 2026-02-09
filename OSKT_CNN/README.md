
## 📊 Dataset Preparation
Replace ``/path/to/data/root`` in the `datasets/DatasetsLoader.py` file with the actual root path to your datasets.
Organize the datasets into the following directory structure:
```
path_to_data_root/
├── CUHK03-NP/
├── Market-1501-v15.09.15/
└── MSMT17_V1/
```

## 💻 Usage
### 1. Teacher Model Training
Download pre-trained models from [LUPerson](https://github.com/DengpanFu/LUPerson) and place them in the following directory:
```
OSKT_CNN/
└── output/
    └── lupws_r50.pth
```
Execute `run_train.sh` to train a high-performance teacher model on the source dataset.

### 2. Weight Chain Refinement
Execute `run_refine_weight_chain.sh` to extract and refine the weight chain from the teacher model.

### 3. Student Model Initialization & Downstream Training
- **Single-Scenario Transfer:** Execute `run_sft.sh` to initialize student models using the weight chain and train them on the same dataset as the teacher model.
- **Cross-Scenario Transfer:** Execute `run_da.sh` to initialize student models using the weight chain and train them on different datasets than the teacher model.


## ⚡ Alternative Quick Start Option
For researchers who wish to bypass the first two stages, we provide pre-computed weight chains for immediate use. Download​ our pre-trained weight chains from [this link](https://drive.google.com/drive/folders/11a2IDAcvxKhNuDDlkhuVdG4vkB0g9-Zd) and organize​ files according to the following directory structure:
```
OSKT_CNN/
└── output/
    └── weight_chain/
        ├── 8/               # 8-inplane weight chain
        │   ├── Market1501/   # Market-1501 refined weight chain
        │   │   ├── gene_matcher.json   # Matching configuration
        │   │   └── transformer_120.pth # Weight chain parameters
        │   └── MSMT17_v1/    # MSMT17 refined weight chain
        │       ├── gene_matcher.json
        │       └── transformer_120.pth
        ├── 16/              # 16-inplane weight chain
        │   ├── Market1501/
        │   │   ├── gene_matcher.json
        │   │   └── transformer_120.pth
        │   └── MSMT17_v1/
        │       ├── gene_matcher.json
        │       └── transformer_120.pth
        └── 32/              # 32-inplane weight chain
            ├── Market1501/
            │   ├── gene_matcher.json
            │   └── transformer_120.pth
            └── MSMT17_v1/
                ├── gene_matcher.json
                └── transformer_120.pth
```




