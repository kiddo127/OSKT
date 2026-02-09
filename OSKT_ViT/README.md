

## 💻 Usage

**Note**: Replace ``/path/to/data/root`` in the DatasetsLoader.py file with the actual root path to your datasets.

### 1. Teacher Model Training
Execute `run_train.sh` to train a high-performance teacher model on the source dataset.

### 2. Weight Chain Refinement
Execute `run_refine_weight_chain.sh` to extract and refine the weight chain from the teacher model.

### 3. Student Model Initialization & Downstream Training
- **Single-Scenario Transfer:** Execute `run_sft.sh` to initialize student models using the weight chain and train them on the same dataset as the teacher model.
- **Cross-Scenario Transfer:** Execute `run_da.sh` to initialize student models using the weight chain and train them on different datasets than the teacher model.
