# CLIP Herbal Treatment Recommendation System
## Academic References and Terminology Guide

---

## Table of Contents
1. [Primary Model References](#primary-model-references)
2. [Evaluation Metrics](#evaluation-metrics)
3. [Technical Terminology](#technical-terminology)
4. [Dataset References](#dataset-references)
5. [Implementation Libraries](#implementation-libraries)
6. [Methodology References](#methodology-references)

---

## Primary Model References

### 1. CLIP Model
**Paper:** Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. International Conference on Machine Learning (ICML).

**Citation:**
```
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & 
Sutskever, I. (2021). Learning Transferable Visual Models From Natural Language 
Supervision. In International Conference on Machine Learning (pp. 8748-8763). PMLR.
```

**Key Contributions:**
- Contrastive learning framework for vision-language alignment
- Zero-shot transfer capabilities to downstream tasks
- State-of-the-art performance without task-specific fine-tuning
- Scalable pre-training on 400M image-text pairs

**Model Used:** `openai/clip-vit-base-patch32`
- Vision Encoder: Vision Transformer (ViT-B/32)
- Text Encoder: Transformer language model
- Embedding Dimension: 512-dimensional joint space
- Pre-training: Contrastive learning on web-scale data

---

### 2. Vision Transformer (ViT)
**Paper:** Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR 2021.

**Citation:**
```
Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., 
Unterthiner, T., ... & Houlsby, N. (2021). An Image is Worth 16x16 Words: 
Transformers for Image Recognition at Scale. 
International Conference on Learning Representations (ICLR).
```

**Relevance:** Foundation for CLIP's vision encoder (ViT-B/32 variant)

**Architecture Details:**
- Patch Size: 32×32 pixels
- Image Resolution: 224×224 pixels
- Transformer Layers: 12
- Attention Heads: 12
- Hidden Size: 768

---

### 3. Contrastive Learning
**Reference:** Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). *A Simple Framework for Contrastive Learning of Visual Representations*. ICML 2020.

**Citation:**
```
Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A Simple Framework 
for Contrastive Learning of Visual Representations. 
In International Conference on Machine Learning (pp. 1597-1607). PMLR.
```

**Relevance:** Theoretical foundation for CLIP's training objective

---

## Evaluation Metrics

### 1. Recall@K
**Reference:** Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.

**Definition:** 
The proportion of relevant items found in the top-K retrieved results.

**Formula:**
```
Recall@K = (Number of relevant items in top-K) / (Total number of relevant items)
```

**Implementation in this project:**
```python
# For image-to-text retrieval
recall_at_k = (top_scores[:, :k].max(dim=1)[0] >= threshold).float().mean()
```

**Used Metrics:**
- **Recall@1**: Accuracy of top-1 recommendation
- **Recall@5**: Accuracy within top-5 recommendations
- **Recall@10**: Accuracy within top-10 recommendations

---

### 2. Mean Reciprocal Rank (MRR)
**Reference:** Voorhees, E. M. (1999). *The TREC-8 Question Answering Track Report*. TREC.

**Definition:**
The average of the reciprocal ranks of the first relevant result for each query.

**Formula:**
```
MRR = (1/N) × Σ(1/rank_i)

where:
- N = total number of queries
- rank_i = position of first relevant result for query i
```

**Implementation:**
```python
reciprocal_rank = 1.0 / rank_of_first_match
MRR = mean(reciprocal_ranks)
```

**Interpretation:**
- MRR = 1.0: Perfect ranking (all first results are correct)
- MRR = 0.5: On average, correct result is at position 2
- Higher MRR indicates better ranking quality

---

### 3. Normalized Discounted Cumulative Gain (NDCG)
**Reference:** Järvelin, K., & Kekäläinen, J. (2002). *Cumulative Gain-based Evaluation of IR Techniques*. ACM Transactions on Information Systems, 20(4), 422-446.

**Citation:**
```
Järvelin, K., & Kekäläinen, J. (2002). Cumulative Gain-based Evaluation of IR 
Techniques. ACM Transactions on Information Systems, 20(4), 422-446.
```

**Definition:**
Measures ranking quality by considering both relevance and position of results.

**Formula:**
```
DCG@K = Σ(rel_i / log₂(i + 1))  for i = 1 to K

NDCG@K = DCG@K / IDCG@K

where:
- rel_i = relevance score of item at position i
- IDCG@K = ideal DCG (perfect ranking)
```

**Implementation:**
```python
dcg = sum(top_scores[i, k].item() / np.log2(k + 2) for k in range(K))
```

**Interpretation:**
- NDCG = 1.0: Perfect ranking order
- Higher NDCG indicates better ranking quality
- Penalizes relevant items appearing lower in results

---

### 4. Mean Average Precision (MAP@K)
**Reference:** Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.

**Definition:**
Average of precision values calculated at each relevant item position.

**Formula:**
```
AP = (1/R) × Σ(P(k) × rel(k))  for k = 1 to K

MAP = mean(AP across all queries)

where:
- R = total number of relevant items
- P(k) = precision at position k
- rel(k) = relevance indicator (0 or 1)
```

**Implementation:**
```python
map_at_k = top_scores[:, :k].mean()
```

---

### 5. Cosine Similarity
**Reference:** Salton, G., & McGill, M. J. (1986). *Introduction to Modern Information Retrieval*. McGraw-Hill.

**Definition:**
Measures the cosine of the angle between two non-zero vectors in multi-dimensional space.

**Formula:**
```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)

where:
- A · B = dot product of vectors A and B
- ||A|| = L2 norm (magnitude) of vector A
- ||B|| = L2 norm of vector B
```

**Range:** [-1, 1]
- 1: Vectors are identical in direction
- 0: Vectors are orthogonal (perpendicular)
- -1: Vectors are opposite in direction

**Implementation:**
```python
similarity = image_embeddings @ text_embeddings.T
# Embeddings are L2-normalized, so @ computes cosine similarity
```

---

### 6. Shannon Entropy (Prediction Diversity)
**Reference:** Shannon, C. E. (1948). *A Mathematical Theory of Communication*. Bell System Technical Journal, 27(3), 379-423.

**Citation:**
```
Shannon, C. E. (1948). A Mathematical Theory of Communication. 
Bell System Technical Journal, 27(3), 379-423.
```

**Definition:**
Measures the diversity/randomness of prediction distribution.

**Formula:**
```
H = -Σ(p_i × log₂(p_i))

Normalized Entropy = H / log₂(N)

where:
- p_i = probability of class i
- N = total number of unique classes
```

**Implementation:**
```python
entropy = -sum((count/total) * np.log2(count/total) for count in herb_counts.values())
normalized_entropy = entropy / np.log2(len(herb_counts))
```

**Interpretation:**
- Higher entropy: More diverse predictions
- Lower entropy: More concentrated predictions
- Normalized to [0, 1] range

---

## Technical Terminology

### Deep Learning Components

#### 1. **Zero-Shot Learning**
**Definition:** The ability of a model to perform tasks without task-specific training examples.

**Reference:** Xian, Y., Lampert, C. H., Schiele, B., & Akata, Z. (2018). *Zero-Shot Learning—A Comprehensive Evaluation of the Good, the Bad and the Ugly*. IEEE TPAMI, 41(9), 2251-2265.

**In this project:**
- CLIP performs herb-skin matching without herb-specific training
- Uses pre-learned vision-language alignment
- Generalizes to new disease-herb combinations

---

#### 2. **Contrastive Learning**
**Definition:** A learning paradigm that learns representations by contrasting positive pairs against negative pairs.

**CLIP's Contrastive Objective:**
```
L = -log(exp(sim(I, T⁺)/τ) / Σ exp(sim(I, Tⁿ)/τ))

where:
- I = image embedding
- T⁺ = matching text embedding (positive pair)
- Tⁿ = non-matching text embeddings (negative pairs)
- τ = temperature parameter
- sim = cosine similarity
```

**Reference:** Radford et al. (2021) - CLIP paper

---

#### 3. **Feature Extraction**
**Definition:** The process of transforming raw data into numerical features that machine learning models can process.

**In this project:**
- **Image Features (512-dimensional):** Visual representations from CLIP vision encoder
- **Text Features (512-dimensional):** Semantic representations from CLIP text encoder
- **Joint Embedding Space:** Both modalities mapped to same 512-dim space

**Reference:** Bengio, Y., Courville, A., & Vincent, P. (2013). *Representation Learning: A Review and New Perspectives*. IEEE TPAMI, 35(8), 1798-1828.

---

#### 4. **Embeddings**
**Definition:** Dense vector representations of data (images or text) in a continuous vector space where similar items are positioned closer together.

**Properties:**
- Fixed-dimensional vectors (512-dim in CLIP)
- Learned through contrastive pre-training
- Capture semantic meaning
- Shared space for vision and language

**Reference:** Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality*. NeurIPS.

---

#### 5. **Normalization (L2 Normalization)**
**Definition:** Scaling vectors to unit length while preserving direction.

**Formula:**
```
normalized_vector = vector / ||vector||₂

where ||vector||₂ = sqrt(Σ(x_i²))
```

**Purpose:**
- Makes cosine similarity computation equivalent to dot product
- Removes magnitude bias, focuses on direction
- Standard practice in contrastive learning

**Implementation:**
```python
embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
```

---

#### 6. **Vision Transformer (ViT)**
**Definition:** Transformer architecture adapted for computer vision by treating image patches as tokens.

**Architecture:**
- Splits image into patches (32×32 in ViT-B/32)
- Linear embedding of flattened patches
- Position embeddings added
- Standard transformer encoder
- [CLS] token for image representation

**Reference:** Dosovitskiy et al. (2021) - ViT paper

---

#### 7. **Attention Mechanism**
**Definition:** A mechanism that allows models to focus on relevant parts of input when processing information.

**Reference:** Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). *Attention Is All You Need*. NeurIPS.

**Citation:**
```
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., 
... & Polosukhin, I. (2017). Attention Is All You Need. 
In Advances in Neural Information Processing Systems (pp. 5998-6008).
```

**Relevance:** Core mechanism in both CLIP's vision and text encoders

---

### Data Processing Terms

#### 8. **Batch Processing**
**Definition:** Processing multiple samples simultaneously to improve computational efficiency.

**Batch Sizes in this project:**
- Image encoding: 16 images/batch
- Text encoding: 32 texts/batch

**Benefits:**
- GPU parallelization
- Reduced processing time
- Memory efficiency

---

#### 9. **Train-Test Split**
**Definition:** Dividing dataset into separate subsets for model evaluation.

**Configuration:**
- **Training Set (70%):** Used to build the retrieval database
- **Test Set (30%):** Used to evaluate recommendation performance

**Reference:** Kohavi, R. (1995). *A Study of Cross-validation and Bootstrap for Accuracy Estimation and Model Selection*. IJCAI.

---

### Retrieval System Terms

#### 10. **Information Retrieval (IR)**
**Definition:** Finding relevant information from large collections based on user queries.

**Components in this project:**
- **Query:** Skin condition image
- **Database:** Herb treatment text descriptions
- **Retrieval:** Finding top-K most similar treatments

**Reference:** Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.

---

#### 11. **Top-K Retrieval**
**Definition:** Selecting K items with highest similarity scores from a database.

**Implementation:**
```python
top_k_scores, top_k_indices = torch.topk(similarity_matrix, k=5)
```

**In this project:** K=5 (recommending 5 herb treatments per skin condition)

---

#### 12. **Cross-Modal Retrieval**
**Definition:** Finding relevant items from one modality (text) using queries from another modality (images).

**In this project:**
- **Query Modality:** Skin condition images
- **Database Modality:** Herbal treatment text descriptions
- **Bridge:** CLIP's joint embedding space

**Reference:** Wang, K., Yin, Q., Wang, W., Wu, S., & Wang, L. (2016). *A Comprehensive Survey on Cross-modal Retrieval*. arXiv preprint arXiv:1607.06215.

---

#### 13. **Similarity Matrix**
**Definition:** A matrix containing pairwise similarity scores between all query and database items.

**Structure:**
```
Shape: [N_images × N_texts]
Entry[i,j] = similarity between image_i and text_j
```

**Computation:**
```python
similarity_matrix = image_embeddings @ text_embeddings.T
```

**Properties:**
- Values in [-1, 1] range (cosine similarity)
- Higher values indicate better matches
- Used for ranking and retrieval

---

#### 14. **Retrieval Coverage**
**Definition:** The percentage of database items that appear in any top-K prediction.

**Formula:**
```
Coverage = (Unique items in top-K predictions) / (Total database items) × 100%
```

**Importance:**
- Measures diversity of retrieval system
- Identifies if some items are never retrieved
- Indicates potential bias in recommendations

---

## Dataset References

### 1. SkinCon Dataset
**Original Paper:** Daneshjou, R., et al. (2022). *Disparities in Dermatology AI Performance on a Diverse, Curated Clinical Image Set*. Science Advances, 8(32).

**Citation:**
```
Daneshjou, R., Vodrahalli, K., Liang, W., Novoa, R. A., Jenkins, M., Rotemberg, V., 
... & Zou, J. (2022). Disparities in Dermatology AI Performance on a Diverse, 
Curated Clinical Image Set. Science Advances, 8(32), eabq6147.
```

**Dataset Characteristics:**
- Diverse skin condition images
- Multiple disease categories
- Clinical-grade annotations
- Skin type diversity

---

### 2. Herbal Medicine Database
**Source:** Traditional Chinese Medicine (TCM) and MeSH (Medical Subject Headings) Integration

**MeSH Reference:**
```
National Library of Medicine (US). (2023). Medical Subject Headings (MeSH). 
Retrieved from https://www.nlm.nih.gov/mesh/
```

**Disease Classification:** MeSH Category: "Skin and Connective Tissue Diseases"

**Dataset Fields:**
- Disease_name: Medical condition name
- Herb_en_name: English herb name
- Ingredient_name: Active compounds
- UsePart: Plant parts used for treatment
- Function: Therapeutic effects
- Disease_alias_name: Alternative disease names

---

## Implementation Libraries

### 1. PyTorch
**Reference:** Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS.

**Citation:**
```
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & 
Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep 
Learning Library. Advances in Neural Information Processing Systems, 32.
```

**Version Used:** PyTorch 2.x
**Purpose:** Deep learning framework for model inference and tensor operations

---

### 2. Hugging Face Transformers
**Reference:** Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., ... & Rush, A. M. (2020). *Transformers: State-of-the-Art Natural Language Processing*. EMNLP.

**Citation:**
```
Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., ... & 
Rush, A. M. (2020). Transformers: State-of-the-Art Natural Language Processing. 
In Proceedings of the 2020 Conference on Empirical Methods in Natural Language 
Processing: System Demonstrations (pp. 38-45).
```

**Version Used:** transformers >= 4.40.0
**Components Used:**
- `CLIPProcessor`: Image and text preprocessing
- `CLIPModel`: Pre-trained CLIP model

---

### 3. Scikit-learn
**Reference:** Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825-2830.

**Citation:**
```
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., 
... & Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python. 
Journal of Machine Learning Research, 12(Oct), 2825-2830.
```

**Purpose:** Train-test splitting, statistical computations

---

### 4. PIL (Pillow)
**Reference:** Clark, A. (2015). *Pillow (PIL Fork) Documentation*. Retrieved from https://pillow.readthedocs.io/

**Purpose:** Image loading and preprocessing

---

### 5. NumPy & Pandas
**NumPy Reference:** Harris, C. R., et al. (2020). *Array Programming with NumPy*. Nature, 585(7825), 357-362.

**Pandas Reference:** McKinney, W. (2010). *Data Structures for Statistical Computing in Python*. SciPy.

**Purpose:** Numerical computing and data manipulation

---

### 6. Matplotlib
**Reference:** Hunter, J. D. (2007). *Matplotlib: A 2D Graphics Environment*. Computing in Science & Engineering, 9(3), 90-95.

**Citation:**
```
Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. 
Computing in Science & Engineering, 9(3), 90-95.
```

**Purpose:** Visualization of evaluation metrics and results

---

## Methodology References

### 1. Vision-Language Pre-training
**Reference:** Li, L. H., Yatskar, M., Yin, D., Hsieh, C. J., & Chang, K. W. (2019). *VisualBERT: A Simple and Performant Baseline for Vision and Language*. arXiv preprint arXiv:1908.03557.

**Relevance:** Pioneering work in joint vision-language learning

---

### 2. Feature Normalization for Similarity
**Reference:** Schroff, F., Kalenichenko, D., & Philbin, J. (2015). *FaceNet: A Unified Embedding for Face Recognition and Clustering*. CVPR 2015.

**Citation:**
```
Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A Unified Embedding 
for Face Recognition and Clustering. 
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition 
(pp. 815-823).
```

**Technique:** L2 normalization before similarity computation
**Benefit:** Makes cosine similarity equivalent to dot product

---

### 3. Multi-Modal Learning
**Reference:** Baltrusaitis, T., Ahuja, C., & Morency, L. P. (2018). *Multimodal Machine Learning: A Survey and Taxonomy*. IEEE TPAMI, 41(2), 423-443.

**Citation:**
```
Baltrusaitis, T., Ahuja, C., & Morency, L. P. (2018). Multimodal Machine Learning: 
A Survey and Taxonomy. IEEE Transactions on Pattern Analysis and Machine Intelligence, 
41(2), 423-443.
```

**Relevance:** Foundation for vision-language integration in CLIP

---

### 4. Medical Image Analysis
**Reference:** Litjens, G., et al. (2017). *A Survey on Deep Learning in Medical Image Analysis*. Medical Image Analysis, 42, 60-88.

**Citation:**
```
Litjens, G., Kooi, T., Bejnordi, B. E., Setio, A. A. A., Ciompi, F., Ghafoorian, M., 
... & Sánchez, C. I. (2017). A Survey on Deep Learning in Medical Image Analysis. 
Medical Image Analysis, 42, 60-88.
```

**Relevance:** Context for skin disease image analysis

---

### 5. Traditional Medicine Integration
**Reference:** Yuan, H., Ma, Q., Ye, L., & Piao, G. (2016). *The Traditional Medicine and Modern Medicine from Natural Products*. Molecules, 21(5), 559.

**Citation:**
```
Yuan, H., Ma, Q., Ye, L., & Piao, G. (2016). The Traditional Medicine and Modern 
Medicine from Natural Products. Molecules, 21(5), 559.
```

**Relevance:** Scientific basis for herbal medicine recommendations

---

### 6. Explainable AI in Healthcare
**Reference:** Holzinger, A., Biemann, C., Pattichis, C. S., & Kell, D. B. (2017). *What Do We Need to Build Explainable AI Systems for the Medical Domain?* arXiv preprint arXiv:1712.09923.

**Relevance:** Importance of interpretable recommendations in medical applications

---

## Project-Specific Methodology

### Data Processing Pipeline
1. **Data Loading:** CSV parsing with pandas
2. **Disease Filtering:** Extract only "Skin and Connective Tissue Diseases"
3. **Text Description Generation:** Create herb-based descriptions
4. **Image Path Verification:** Recursive search with disease subdirectories
5. **Feature Extraction:** Batch processing with CLIP encoders
6. **Similarity Computation:** Cosine similarity via normalized dot product
7. **Top-K Retrieval:** PyTorch topk operation
8. **Evaluation:** Multiple metrics (Recall@K, MRR, NDCG, MAP, Entropy)

### Key Design Decisions

#### 1. Zero-Shot Approach
**Decision:** Use pre-trained CLIP without fine-tuning
**Rationale:** 
- Leverages CLIP's strong zero-shot transfer capabilities
- No need for herb-specific training data
- Generalizes to new disease-herb combinations

**Reference:** Radford et al. (2021) demonstrated CLIP's zero-shot effectiveness

#### 2. Joint Embedding Space
**Challenge:** Mapping images and text to same semantic space
**Solution:** CLIP's contrastively learned joint embedding (512-dim)
**Benefit:** Direct similarity computation via dot product

#### 3. Batch Size Selection
**Image Batch:** 16 (balanced GPU memory and processing speed)
**Text Batch:** 32 (text processing is less memory-intensive)

#### 4. Top-K Selection
**K=5 chosen based on:**
- Clinical utility (multiple treatment options)
- Evaluation standard in retrieval literature
- Balance between diversity and relevance

---

## Evaluation Framework

### Standard Vision-Language Retrieval Metrics

**Primary Metrics:**
1. **Recall@K** - Retrieval accuracy
2. **MRR** - Ranking quality
3. **NDCG@K** - Position-aware relevance
4. **MAP@K** - Average precision

**Auxiliary Metrics:**
1. **Similarity Score Distribution** - Overall quality
2. **Confidence Distribution** - Prediction reliability
3. **Coverage** - Database utilization
4. **Entropy** - Prediction diversity
5. **Per-Disease Performance** - Class-specific analysis

**Reference:** Standard evaluation protocol from:
- CLIP paper (Radford et al., 2021)
- COCO retrieval benchmark
- Flickr30K retrieval benchmark

---

## Citation Template for This Project

```
[Your Name]. (2025). CLIP-based Herbal Treatment Recommendation System for Skin Diseases: 
Zero-Shot Cross-Modal Retrieval Approach. [Academic Institution]. 
Implementation based on Radford et al. (2021) CLIP architecture.
```

---

## Acknowledgments

This implementation builds upon:
- **CLIP Model:** OpenAI Research
- **Hugging Face:** Model hosting and transformers library
- **SkinCon Dataset:** Stanford University research team
- **PyTorch:** Meta AI Research
- **Open-Source Community:** Various libraries and tools

---

## Disclaimer

This system is designed for **academic research purposes only**. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical conditions.

---

## Version Information

- **Document Version:** 1.0
- **Last Updated:** November 24, 2025
- **CLIP Model Version:** openai/clip-vit-base-patch32
- **Code Implementation:** Python 3.8+
- **Primary Framework:** PyTorch 2.x

---

## Contact for Academic Inquiries

For questions about this implementation or academic collaboration:
- Review the original CLIP paper: arXiv:2103.00020
- Check Hugging Face model card: https://huggingface.co/openai/clip-vit-base-patch32
- Consult course instructor or research advisor

---

*This document provides comprehensive academic references for the CLIP Herbal Treatment Recommendation System project. All citations should be verified and updated according to your institution's citation guidelines.*
