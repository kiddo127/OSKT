# One-Shot Knowledge Transfer for Scalable Person Re-Identification (ICCV 2025)
[PyTorch] Official implementation of "One-Shot Knowledge Transfer for Scalable Person Re-Identification"​ (ICCV 2025)

Official PyTorch implementation of [One-Shot Knowledge Transfer for Scalable Person Re-Identification](https://openaccess.thecvf.com/content/ICCV2025/html/Li_One-Shot_Knowledge_Transfer_for_Scalable_Person_Re-Identification_ICCV_2025_paper.html) (ICCV 2025), a novel knowledge inheritance approach that enables efficient generation of scalable person re-identification models through one-shot weight chain refinement.

## 📖 Overview
OSKT addresses the challenge of repetitive computations in traditional model compression methods by introducing an intermediate knowledge carrier called weight chain. This approach allows efficient generation of student models with varying sizes through a single computation, making it ideal for edge computing scenarios with diverse resource constraints.


<img width="4487" height="3355" alt="overview" src="https://github.com/user-attachments/assets/5d84cb88-46f2-4de7-87fa-1ee90de8fad6" />
<img width="4487" height="3355" alt="overview" src="https://github.com/user-attachments/assets/1e996360-b909-4e9e-a29d-16db806ec048" />


## ✨ Key Features
- **One-shot Knowledge Transfer**: Generate multiple student models with a single computation
- **Architecture Agnostic**: Compatible with both CNN and ViT architectures
- **Scalable Weight Chains**: Support for multiple weight chains covering wide parameter ranges
- **State-of-the-Art Performance**: Outperforms traditional knowledge distillation and pruning methods

## 🏗️ Model Architectures
Our method leverages the refined weight chain​ to generate a dense spectrum of student models​ spanning from the weight chain's own width up to the full width of the teacher model. To evaluate the effectiveness of our approach, we construct the following series of models for performance assessment.
### ResNet Model Series
| Model | Role | Inplanes | Multipliers | Params | FLOPs |
|-------|-----|----------|-------------|--------|-------|
| Res-50-S1 | Student | 8 | [2,4,8] | 0.37M | 0.05G |
| Res-50-S2 | Student | 16 | [2,3,4] | 0.61M | 0.14G |
| Res-50-S3 | Student | 16 | [2,4,8] | 1.48M | 0.19G |
| Res-50-S4 | Student | 32 | [2,3,4] | 2.42M | 0.51G |
| Res-50-S5 | Student | 32 | [2,4,8] | 5.89M | 0.70G |
| Res-50-S6 | Student | 64 | [2,3,4] | 9.64M | 1.92G |
| ResNet50 | Teacher | 64 | [2,4,8] | 23.5M | 2.70G |

### ViT Model Series
| Model | Role | Dimension | Params | MACs |
|-------|------|-----------|--------|------|
| ViT-S-S1 | Student | 120 | 2.20M | 0.29G |
| ViT-S-S2 | Student | 168 | 4.42M | 0.56G |
| ViT-S | Teacher | 384 | 21.74M | 2.87G |
| ViT-B-S1 | Student | 264 | 10.35M | 1.36G |
| ViT-B-S2 | Student | 336 | 16.68M | 2.20G |
| ViT-B | Teacher | 768 | 86.24M | 11.37G |


## 📊 Experimental Results
We evaluate our method under two distinct settings: Knowledge Transfer in a Single Scenario​ (where weight chain refinement and student model training share the same dataset) and Knowledge Transfer across Scenarios​ (where these two stages utilize different datasets). Our comparative analysis encompasses four initialization strategies for student models: random initialization (Scratch), parameter selection from the teacher model (Weight Selection, WTSel), knowledge distillation (KD++), and model pruning (DepGraph). The performance of student models after downstream training is reported in the format of mAP(%)/Rank-1(%).

**Note**:​ Our method demonstrates substantial performance advantages​ over competing compression approaches when generating student models that precisely match the width of the weight chain (e.g., Res-50-S1, Res-50-S3, Res-50-S5, ViT-S-S1, ViT-B-S1). The performance gap narrows when producing wider student models through efficient expansion. For applications where scalability is not required, our approach serves as an exceptional stand-alone compression technique.

### Knowledge Transfer in a Single Scenario
| Method | Res-50-S1 | Res-50-S3 | Res-50-S5 |
|--------|-------------------|-------------------|-------------------|
| Scratch | 30.6/53.3 | 47.4/70.9 | 60.7/80.9 |
| WTSel | 48.5/72.3 | 63.0/81.5 | 84.1/93.0 |
| KD++ | 41.5/65.7 | 59.9/80.4 | 71.4/86.9 |
| DepGraph | 61.3/80.7 | 83.2/92.7 | 86.3/94.3 |
| **OSKT** | **75.7/89.4** | **84.7/93.3** | **87.6/94.5** |

| Method | ViT-S-S1 | ViT-S-S2 | ViT-B-S1 | ViT-B-S2 |
|--------|----------|----------|----------|----------|
| Scratch | 13.9/23.9 | 18.3/31.2 | 22.2/37.4 | 21.9/36.3 |
| WTSel | 41.0/60.5 | 54.8/75.1 | 16.8/31.7 | 58.7/78.2 |
| KD++ | 22.6/38.7 | 25.0/41.2 | 25.9/41.7 | 28.2/44.4 |
| DepGraph | 56.5/74.1 | 69.0/84.2 | 15.3/30.1 | 81.5/91.7 |
| **OSKT** | **74.2/87.1** | **77.2/89.0** | **81.6/91.9** | **82.9/92.4** |


### Knowledge Transfer across Scenarios


| Scenario | Method | Res-50-S1 | Res-50-S2 | Res-50-S3 | Res-50-S4 | Res-50-S5 | Res-50-S6 |
|----------|---------|------------|------------|------------|------------|------------|------------|
| **MS→M** | WTSel | 41.3/64.0 | 54.6/76.5 | 60.1/80.2 | 73.1/88.5 | 77.4/90.7 | **86.1**/94.0 |
| | KD++ | 54.6/76.7 | 63.2/82.5 | 71.2/87.5 | 77.5/90.5 | 78.9/91.0 | 80.9/92.2 |
| | DepGraph | 61.6/81.0 | **75.0/89.2** | 77.3/90.3 | 82.4/**92.8** | 82.3/92.7 | 82.1/92.5 |
| | **OSKT** | **72.3/87.5** | 74.4/89.6 | **82.6/92.6** | **82.9**/92.7 | **85.6/93.9** | **86.1/94.1** |
| **MS→C** | WTSel | 16.6/15.5 | 24.8/24.6 | 22.2/21.3 | 38.9/40.1 | 45.1/45.5 | 68.9/**71.8** |
| | KD++ | 26.8/27.0 | 34.3/35.4 | 41.6/42.9 | 52.0/53.3 | 53.6/55.1 | 56.1/57.6 |
| | DepGraph | 31.3/31.7 | 49.0/51.5 | 54.8/57.8 | 62.4/64.4 | 65.0/68.7 | 65.8/69.9 |
| | **OSKT** | **45.7/47.3** | **49.2/52.2** | **62.5/65.1** | **63.1/65.8** | **68.6/71.7** | **69.8**/71.6 |
| **M→C** | WTSel | 18.0/15.9 | 27.4/27.1 | 28.0/27.9 | 45.3/47.4 | 55.0/56.9 | 65.7/68.3 |
| | KD++ | 21.1/21.1 | 31.8/32.1 | 37.5/37.5 | 46.1/47.3 | 50.3/52.1 | 55.9/58.4 |
| | DepGraph | 23.4/22.7 | 47.7/49.9 | 53.4/56.4 | **65.1/68.2** | 65.8/68.3 | 67.0/70.5 |
| | **OSKT** | **44.6/47.4** | **47.9/50.9** | **60.9/64.1** | 63.5/66.5 | **68.0/70.6** | **69.5/72.3** |


| Scenario | Method | ViT-S-S1 | ViT-S-S2 | ViT-B-S1 | ViT-B-S2 |
|----------|--------|----------|----------|----------|----------|
| **MS→M** | WTSel | 42.8/62.9 | 55.4/75.3 | 15.8/30.5 | 60.0/79.1 |
| | KD++ | 38.1/55.9 | 41.2/60.7 | 40.2/59.3 | 40.7/59.4 |
| | DepGraph | 65.5/82.1 | 74.6/86.9 | 63.7/80.7 | **83.3/92.1** |
| | **OSKT** | **76.5/88.7** | **79.0/89.7** | **82.1/91.9** | 83.0/92.0 |
| **MS→C** | WTSel | 1.6/0.9 | 20.9/19.9 | 3.0/2.5 | 24.2/23.1 |
| | KD++ | 15.8/14.1 | 17.9/16.6 | 18.4/17.1 | 20.5/18.7 |
| | DepGraph | 46.7/47.9 | 59.4/61.3 | 41.1/43.1 | **71.0/74.5** |
| | **OSKT** | **61.2/63.5** | **63.9/66.4** | **69.4/72.4** | 70.3/73.0 |
| **M→C** | WTSel | 1.7/1.4 | 20.8/19.5 | 2.5/1.9 | 5.2/4.5 |
| | KD++ | 10.4/9.0 | 11.6/10.4 | 12.2/10.1 | 13.9/12.1 |
| | DepGraph | 40.3/41.4 | **51.7**/52.7 | 6.2/5.2 | **63.1/65.6** |
| | **OSKT** | **49.3/50.9** | 51.3/**52.9** | **59.4/61.7** | 61.9/64.4 |


## 💻 Usage

**Note**: Replace ``/path/to/data/root`` in the DatasetsLoader.py file with the actual root path to your datasets.

### 1. Teacher Model Training
### 2. Weight Chain Refinement
### 3. Student Model Initialization & Downstream Training

