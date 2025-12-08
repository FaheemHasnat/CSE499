# Vision-Language Models for Dermatological Treatment Recommendations

A comprehensive implementation of vision-language models (CLIP, BLIP-2, SigLIP) for cross-modal retrieval, matching skin condition images with herbal treatment descriptions.

## Overview

This project leverages state-of-the-art vision-language models to create an intelligent recommendation system that:
- Analyzes dermatological condition images
- Retrieves relevant herbal treatment descriptions
- Evaluates retrieval performance using multiple metrics

## Models Implemented

### 1. CLIP (Contrastive Language-Image Pre-training)
- **Model**: `openai/clip-vit-base-patch32`
- **Architecture**: Dual encoder with ViT-B/32 vision encoder and transformer text encoder
- **Embedding Dimension**: 512
- **Loss Function**: InfoNCE (contrastive loss)

### 2. BLIP-2 (Bootstrapping Language-Image Pre-training)
- **Model**: `Salesforce/blip2-opt-2.7b`
- **Architecture**: Q-Former bridge between frozen vision encoder and OPT-2.7B language model
- **Image Embedding**: 1,408 dimensions
- **Text Embedding**: 2,560 dimensions

### 3. SigLIP (Sigmoid Loss for Language-Image Pre-training)
- **Model**: `google/siglip-base-patch16-224`
- **Architecture**: Vision-language model with sigmoid-based pairwise loss
- **Embedding Dimension**: 768
- **Total Parameters**: 203.16M

## Project Structure

```
499A/
├── config.py                          # Configuration settings
├── main.py                            # Main application entry point
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .env.template                      # Environment variables template
├── .gitignore                         # Git ignore rules
├── data/
│   ├── herb2_final_clean.csv         # Herbal treatment database
│   └── skincon_preprocessed.csv      # Skin condition metadata
├── main/
│   ├── DataPreprocessing.ipynb       # Data preprocessing pipeline
│   ├── DermnetCLIP.ipynb            # CLIP experiments (Dermnet dataset)
│   ├── DermnetSigLIP.ipynb          # SigLIP experiments (Dermnet dataset)
│   ├── skincon_herb_clip.ipynb      # CLIP-based retrieval (Skincon dataset)
│   ├── skincon_herb_blip2.ipynb     # BLIP-2 based retrieval (Skincon dataset)
│   └── skincon-herb-siglip.ipynb    # SigLIP-based retrieval (Skincon dataset)
├── support/
│   ├── __init__.py
│   ├── image_processor.py            # Image processing utilities
│   ├── model_handler.py              # Model loading and inference
│   ├── plant_database.py             # Plant/herb database management
│   └── utils.py                      # General utilities
├── result/
│   ├── blip2_test_split.csv
│   ├── blip2_train_split.csv
│   ├── skincon_herb_clip_results_SKIN_ONLY (1).csv
│   ├── CLIP_References_and_Terminology.md
│   ├── BLIP2_References_and_Terminology.md
│   ├── clip_evaluation_metrics.png
│   ├── blip2_herb_recommendation_*.png
│   └── skin_herb_match_*.png
└── others/
    ├── 499A_Literature-Review.pptx
    ├── CSE499A_project.proposal.pdf
    ├── Details.about.Preprocessing.pdf
    ├── Presentation-Vision-Language-Model.pptx
    ├── Proposed.Solution.1.docx
    └── Update*.pdf                    # Project progress updates
```

**Note**: Dataset preprocessing steps were presented in the First Week Update documentation (available in `others/` directory).

## Dataset

### Skin Condition Dataset
- **Source**: Skincon preprocessed dataset
- **Content**: Dermatological condition images with metadata
- **Diseases Covered**: 36 unique skin and connective tissue diseases
- **Total Samples**: 6,337 images

### Herbal Treatment Dataset
- **Source**: Herb2 final clean dataset
- **Content**: Medicinal plant compound information and therapeutic functions
- **Description**: Detailed text descriptions of herbal treatments

## Methodology

### 1. Data Preprocessing
- Image normalization and resizing
- Text tokenization and preprocessing
- Train-test stratified split (70-30 ratio)

### 2. Embedding Extraction
- Pre-compute image and text embeddings using pretrained models
- L2 normalization for cosine similarity computation
- Batch processing for computational efficiency

### 3. Cross-Modal Retrieval
- Compute similarity matrix between test images and training texts
- Retrieve top-K most similar herbal treatments
- Evaluate using multiple metrics

### 4. Evaluation Metrics
- **Recall@K** (K=1,3,5): Proportion of correct matches in top-K
- **Mean Reciprocal Rank (MRR)**: Average reciprocal rank of first relevant result
- **Mean Average Precision (MAP@K)**: Average precision across queries
- **Normalized DCG (NDCG@K)**: Ranking quality with position discount
- **Shannon Entropy**: Diversity of recommendations

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

1. Clone the repository:
```bash
git clone https://github.com/FaheemHasnat/CSE499.git
cd CSE499
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pretrained models (automatic on first run):
- CLIP: `openai/clip-vit-base-patch32`
- BLIP-2: `Salesforce/blip2-opt-2.7b`
- SigLIP: `google/siglip-base-patch16-224`

## Usage

### Running Notebooks

Launch Jupyter and open desired notebook:
```bash
jupyter notebook main/skincon_herb_clip.ipynb
```

### Using Main Script

```python
from support.model_handler import VLMModelHandler
from support.image_processor import ImageProcessor
from config import MODEL_CONFIG, IMAGE_CONFIG

# Initialize model
handler = VLMModelHandler(MODEL_CONFIG)
handler.load_model()

# Process image
processor = ImageProcessor(IMAGE_CONFIG)
image = processor.load_image('path/to/skin_condition.jpg')

# Get recommendations
recommendations = handler.analyze_image(image)
```

## Training Configuration

### SigLIP Training Hyperparameters
- **Optimizer**: AdamW
- **Learning Rate**: 1e-5
- **Weight Decay**: 0.01
- **Batch Size**: 32
- **Epochs**: 5
- **Warmup Steps**: 100
- **Gradient Clipping**: 1.0

## Mathematical Foundations

For detailed mathematical formulations, loss functions, and evaluation metrics, see the mathematical documentation in the [Wiki](https://github.com/FaheemHasnat/CSE499/wiki).

Key equations:

**Cosine Similarity**:
```
similarity(v, t) = (v · t) / (||v||₂ ||t||₂)
```

**CLIP Loss (InfoNCE)**:
```
L_CLIP = -1/(2N) Σ [log(exp(s_ii/τ)/Σexp(s_ij/τ)) + log(exp(s_ii/τ)/Σexp(s_ji/τ))]
```

**SigLIP Loss**:
```
L_SigLIP = -1/N Σ log(σ(z_ij · y_ij))
```

## Results

Performance varies by model and configuration. Typical results:
- **Recall@1**: 0.15-0.35
- **Recall@5**: 0.40-0.65
- **MRR**: 0.25-0.45
- **NDCG@5**: 0.35-0.55

## Datasets & Results

All datasets and experiment results are stored here:  
https://drive.google.com/drive/folders/1N4v7PnTyxt3qKfpi5gD38N2PtlVNDTCr

## Video Demonstration

The project demo video can be found here:  
https://drive.google.com/file/d/1hdTOgkEcyhsEm4_pCOp7N-kpeQB2NETa/view?usp=drive_link

## Contributors & Branch Responsibilities

### Emon Hossen
- **Responsible for**: `Emon-Hossen` branch
- **Work done**: Skincon_herb_clip & Skincon_herb_blip2

### Faheem Hasnat
- **Responsible for**: `Branch1` branch
- **Work done**: DataPreprocessing & Dermnet_herb_blip2

### Kazi Tanora Akther
- **Responsible for**: `Kazi-Tanora` and `Kazi-Tanora-2` branches
- **Work done**: Dermnet_Herb_siglip & skincon_herb_siglip

**Note**: Additional necessary files and documentation are available in the [Wiki](https://github.com/FaheemHasnat/CSE499/wiki).

## References

1. Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML*.
2. Li, J., et al. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. *arXiv:2301.12597*.
3. Zhai, X., et al. (2023). Sigmoid Loss for Language Image Pre-training. *ICCV*.


## Acknowledgments

- Hugging Face Transformers library
- OpenAI CLIP
- Salesforce BLIP-2
- Google SigLIP
