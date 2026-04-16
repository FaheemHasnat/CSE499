# Herbal Treatment Recommendation System for Skin Diseases

## Project Overview

This project implements multimodal deep learning models for recommending herbal treatments for skin diseases by analyzing skin condition images and matching them with traditional medicine compounds. The research explores two distinct approaches:

1. **Baseline Supervised Model**: A fine-tuned multimodal architecture using EfficientNetB0 for image encoding and TF-IDF for text encoding, achieving supervised multi-label prediction.
2. **Vision-Language Models**: Zero-shot and few-shot learning approaches using CLIP and BLIP-2 pre-trained models for cross-modal retrieval.

### Problem Statement
Skin diseases often require treatment with multiple therapeutic compounds/herbs. This project develops automated systems to predict relevant compounds for given skin disease images, enabling intelligent recommendation systems for dermatology applications.

---

## Research Approaches & Models Implemented

### Approach 1: Supervised Multimodal Learning (Baseline)
**Model**: Custom EfficientNetB0 + TF-IDF Fusion Architecture
- **Image Encoder**: EfficientNetB0 (pretrained on ImageNet) → 1,280-dim features
- **Text Encoder**: TF-IDF embeddings (128-dim) from compound names
- **Fusion Strategy**: Early concatenation followed by classification head
- **Task**: Multi-label image classification (554 therapeutic compounds)
- **Implementation**: [499B/cse499b-skincon-herb-baseline-model.ipynb](499B/cse499b-skincon-herb-baseline-model.ipynb)

### Approach 2: Zero-Shot Cross-Modal Retrieval (CLIP)
**Model**: `openai/clip-vit-base-patch32`
- **Embedding Space**: 512-dimensional joint image-text space
- **Approach**: Zero-shot cross-modal retrieval without fine-tuning
- **Key Features**:
  - Fast inference
  - Lower memory requirements (~6GB VRAM)
  - Generalizable to unseen compounds
- **Implementation**: [notebooks/skincon_herb_clip.ipynb](notebooks/skincon_herb_clip.ipynb)
- **Evaluation Metrics**: Recall@K, MRR, NDCG, Cosine Similarity

### Approach 3: Advanced Vision-Language Understanding (BLIP-2)
**Model**: `Salesforce/blip2-opt-2.7b`
- **Architecture**: Q-Former with frozen vision/text encoders
- **Parameters**: ~2.7 billion
- **Approach**: Query-based bridging for vision-language understanding
- **Key Features**:
  - Advanced multimodal reasoning
  - Separate train/test evaluation metrics
  - Superior accuracy for complex semantic tasks
- **Implementation**: [notebooks/skincon_herb_blip2.ipynb](notebooks/skincon_herb_blip2.ipynb)

---

## Dataset Information

### Dataset Specifications
- **Total Samples**: 9,601 dermatology images
- **Training Set**: 6,720 samples (70%)
- **Test Set**: 2,881 samples (30%)
- **Image Format**: RGB images 224×224 pixels
- **Source**: SKINCON dataset (diverse skin conditions)
- **Target Classes**: 554 therapeutic compounds
- **Label Type**: Multi-hot binary vectors (extreme class imbalance ~99.7% zeros)

### Data Components
1. **SkinCon Dataset**
   - Diverse, clinically-annotated skin condition images
   - Multiple disease categories
   - High-quality image annotations

2. **Herbal Medicine Database** (`herb2_final_clean.csv`)
   - Traditional Chinese Medicine (TCM) compounds
   - MeSH disease classification
   - Focused on "Skin and Connective Tissue Diseases"
   - 554 unique therapeutic compounds/herbs

3. **Supporting Data Structures**
   - `label_encoder.pkl`: Compound name to index mapping
   - `disease_compound_mapping.json`: Manual disease-to-compound mappings
   - `vectorizer.pkl`: TF-IDF vectorizer for text encoding

### Label Statistics
- **Mean compounds per image (train)**: 1.64 compounds
- **Mean compounds per image (test)**: 1.63 compounds
- **Sparsity**: 99.7% zeros in label matrix
- **Class Distribution**: Highly imbalanced (compound occurrences range from 1-2 to hundreds)

---

## Baseline Model Architecture

### Multimodal Fusion Design

```
Input: (Image, Compound Labels, Text Features)
  │
  ├─ Image Branch (EfficientNetB0)
  │   ├─ Pretrained: ImageNet weights
  │   ├─ Fine-tuned: Yes
  │   └─ Output: 1,280-dimensional features
  │
  ├─ Text Branch (TF-IDF → Dense Encoding)
  │   ├─ TF-IDF vectorizer (character 2-3 grams)
  │   ├─ Max features: 128
  │   └─ Output: 128-dimensional embeddings
  │
  └─ Fusion & Classification Layers
      ├─ Concatenate: [1280 + 128] = 1408 dimensions
      ├─ Dense(1408 → 512) + BatchNorm + ReLU + Dropout(0.3)
      ├─ Dense(512 → 256) + BatchNorm + ReLU + Dropout(0.3)
      └─ Dense(256 → 554) Output logits for 554 compounds
```

### Training Configuration

#### Loss & Optimization
- **Loss Function**: BCEWithLogitsLoss (Multi-label binary classification)
- **Class Weighting**: Applied to handle 99% sparsity
- **Optimizer**: AdamW (lr=1×10⁻⁴, weight_decay=1×10⁻⁵)
- **Schedule**: ReduceLROnPlateau (factor=0.5, patience=1 epoch)

#### Hyperparameters
- **Batch Size**: 32 (single split), 16 (cross-validation)
- **Epochs**: 20 (single split), 10 per fold (cross-validation)
- **Gradient Clipping**: max_norm=1.0
- **Decision Threshold**: 0.3 (logits → binary predictions)
- **Early Stopping**: Patience=100 epochs
- **Total Parameters**: 5,020,710 (~5.02M)

#### Image Augmentation
- Resize: 224×224
- RandomHorizontalFlip (p=0.5)
- ColorJitter (brightness/contrast/saturation ±0.2)
- ImageNet normalization

---

## Project Structure

```
.
├── 499B/                                          # Final project deliverables
│   ├── cse499b-skincon-herb-baseline-model.ipynb # Supervised learning implementation
│   ├── cse499b-skincon-herb-final-training.ipynb # Final training pipeline
│   ├── CSE499B_SKINCON_HERB_BASELINE_MODEL_DOCUMENTATION.md
│   └── results/                                   # Training results and metrics
│
├── data/                                          # Datasets
│   ├── herb2_final_clean.csv                     # Herbal compound database
│   └── skincon_preprocessed.csv                  # Preprocessed SkinCon dataset
│
├── models/                                        # Model documentation & references
│   ├── BLIP2_References_and_Terminology.md
│   └── CLIP_References_and_Terminology.md
│
├── notebooks/                                     # Vision-language model implementations
│   ├── skincon_herb_clip.ipynb                   # CLIP zero-shot approach
│   └── skincon_herb_blip2.ipynb                  # BLIP-2 multimodal approach
│
├── outputs/                                       # Model predictions & splits
│   ├── blip2_test_split.csv
│   └── blip2_train_split.csv
│
├── results/                                       # Processed data & mappings
│   ├── disease_compound_mapping.json
│   ├── processed_dataset.csv
│   ├── train.csv
│   └── test.csv
│
├── src/                                           # Reusable utilities
│   ├── image_processor.py
│   ├── model_handler.py
│   ├── plant_database.py
│   └── utils.py
│
├── config.py                                      # Configuration settings
├── main.py                                        # Entry point
└── README.md
```

---

## Key Research Contributions

### Baseline Model
- **Multimodal Fusion**: Early concatenation of image and text embeddings
- **Extreme Sparsity Handling**: Custom class weighting for 99%+ zero labels
- **Fine-tuned EfficientNetB0**: Optimized for dermatological image analysis
- **Text Feature Engineering**: TF-IDF character n-grams for compound semantics

### Vision-Language Model Comparisons
- **Computational Efficiency**: CLIP vs BLIP-2 inference time analysis
- **Zero-Shot Generalization**: CLIP's ability on unseen compounds
- **Semantic Understanding**: BLIP-2's advanced reasoning capabilities
- **Evaluation Framework**: Comprehensive metrics (Recall@K, MRR, NDCG)

---

## Evaluation Metrics & Results

### Metrics Used
- **Recall@K**: Percentage of true compounds in top-K recommendations
- **Mean Reciprocal Rank (MRR)**: Average reciprocal rank of first correct compound
- **NDCG (Normalized Discounted Cumulative Gain)**: Ranking quality considering position
- **Cosine Similarity**: Direct embedding space similarity scores

### Model Comparison Framework
- Train/test split evaluation for supervised models
- Zero-shot evaluation for pre-trained models
- Cross-validation for robustness assessment
- Multiple decision thresholds for precision-recall tradeoffs

---

## Requirements

```
Python 3.8+
PyTorch 2.0+
Transformers 4.40+
scikit-learn
pandas
Pillow
numpy
CUDA-capable GPU (recommended for training)
```

---

## Usage Instructions

### For Supervised Baseline Model
1. Navigate to `499B/` folder
2. Run `cse499b-skincon-herb-baseline-model.ipynb` for model training
3. Results saved in `499B/results/`

### For Vision-Language Models
1. Run `notebooks/skincon_herb_clip.ipynb` for CLIP evaluation
2. Run `notebooks/skincon_herb_blip2.ipynb` for BLIP-2 evaluation
3. Results saved in `outputs/` directory

### For Custom Workflows
- Use utilities in `src/` for image processing and model handling
- Modify `config.py` for custom hyperparameters
- Execute `main.py` for integrated pipeline

---

## Academic References

For detailed references, terminology, and citations, see:
- [models/CLIP_References_and_Terminology.md](models/CLIP_References_and_Terminology.md)
- [models/BLIP2_References_and_Terminology.md](models/BLIP2_References_and_Terminology.md)
- [499B/CSE499B_SKINCON_HERB_BASELINE_MODEL_DOCUMENTATION.md](499B/CSE499B_SKINCON_HERB_BASELINE_MODEL_DOCUMENTATION.md)

---

## Project Status & Deliverables

✅ **Completed**:
- Baseline supervised multimodal model implementation
- CLIP zero-shot cross-modal retrieval system
- BLIP-2 advanced vision-language model
- Comprehensive data preprocessing pipeline
- Model comparison framework
- Academic documentation and references

**Research Focus**:
- Multimodal learning for medical recommendation systems
- Extreme class imbalance handling in dermatology datasets
- Vision-language model adaptability for specialized domains
- Zero-shot vs. fine-tuned learning comparisons

---

## Contributors

- **Emon Hossen** - University Project CSE499A and CSE499B

---

## Topic

**Academic Research Project** - CSE499A, CSE499B Senior Design I and II  
**Focus**: Vision-Language Models for Herbal Treatment Recommendations in Dermatology  
**Institution**: ECE Department, North South University
