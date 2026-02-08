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


