# CSE499 - Herbal Treatment Recommendation System

## Project Overview
This project implements vision-language models for recommending herbal treatments for skin diseases by matching skin condition images with traditional medicine information.

## Models Implemented

### 1. CLIP (Contrastive Language-Image Pre-training)
- **Model**: `openai/clip-vit-base-patch32`
- **Approach**: Zero-shot cross-modal retrieval
- **Embedding Dimension**: 512-dim joint space
- **Key Features**:
  - Fast inference
  - Lower memory requirements (~6GB VRAM)
  - Good for general retrieval tasks

### 2. BLIP-2 (Bootstrapping Language-Image Pre-training v2)
- **Model**: `Salesforce/blip2-opt-2.7b`
- **Approach**: Q-Former with frozen encoders
- **Parameters**: ~2.7 billion
- **Key Features**:
  - Advanced vision-language understanding
  - Separate train/test evaluation
  - Higher accuracy for complex tasks

## Project Structure

```
├── data/
│   ├── herb2_final_clean.csv          # Herbal medicine database
│   └── skincon_preprocessed.csv       # Skin condition dataset
├── models/
│   ├── BLIP2_References_and_Terminology.md
│   └── CLIP_References_and_Terminology.md
├── notebooks/
│   ├── skincon_herb_clip.ipynb        # CLIP implementation
│   └── skincon_herb_blip2.ipynb       # BLIP-2 implementation
└── outputs/
    ├── blip2_test_split.csv
    └── blip2_train_split.csv
```

## Datasets

### SkinCon Dataset
- Diverse skin condition images
- Multiple disease categories
- Clinical-grade annotations

### Herbal Medicine Database
- Traditional Chinese Medicine (TCM) compounds
- MeSH disease classification
- Focused on "Skin and Connective Tissue Diseases"

## Key Features

- **Cross-Modal Retrieval**: Match images to text descriptions
- **Top-K Recommendations**: Provide multiple treatment options
- **Comprehensive Evaluation**: Recall@K, MRR, NDCG metrics
- **Academic References**: Complete citations and terminology guide

## Evaluation Metrics

- **Recall@K**: Accuracy within top-K results
- **Mean Reciprocal Rank (MRR)**: Ranking quality
- **NDCG**: Position-aware evaluation
- **Similarity Scores**: Cosine similarity in embedding space

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.40+
- CUDA-capable GPU (recommended)

## Usage

1. Mount Google Drive with datasets
2. Run either notebook:
   - `skincon_herb_clip.ipynb` for CLIP model
   - `skincon_herb_blip2.ipynb` for BLIP-2 model
3. Results saved to `outputs/` directory

## References

See detailed academic references in:
- `models/CLIP_References_and_Terminology.md`
- `models/BLIP2_References_and_Terminology.md`

## Contributors

- Emon Hossen

## License

Academic research project - CSE499
