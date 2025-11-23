# BLIP-2 Herbal Treatment Recommendation System
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

### 1. BLIP-2 Model
**Paper:** Li, J., Li, D., Savarese, S., & Hoi, S. (2023). *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*. arXiv preprint arXiv:2301.12597.

**Citation:**
```
Li, J., Li, D., Savarese, S., & Hoi, S. (2023). BLIP-2: Bootstrapping Language-Image 
Pre-training with Frozen Image Encoders and Large Language Models. 
arXiv preprint arXiv:2301.12597.
```

**Key Contributions:**
- Q-Former architecture for vision-language alignment
- Efficient training with frozen vision and language models
- State-of-the-art performance on vision-language tasks

**Model Used:** `Salesforce/blip2-opt-2.7b`
- Vision Encoder: Pre-trained frozen vision transformer
- Q-Former: 32 learnable query tokens
- Language Model: OPT-2.7B (frozen)
- Total Parameters: ~2.7 billion

---

### 2. Vision Transformer (ViT) Base
**Paper:** Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR 2021.

**Citation:**
```
Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., 
Unterthiner, T., ... & Houlsby, N. (2021). An Image is Worth 16x16 Words: 
Transformers for Image Recognition at Scale. 
International Conference on Learning Representations (ICLR).
```

**Relevance:** Foundation for BLIP-2's vision encoder

---

### 3. OPT Language Model
**Paper:** Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., ... & Zettlemoyer, L. (2022). *OPT: Open Pre-trained Transformer Language Models*. arXiv preprint arXiv:2205.01068.

**Citation:**
```
Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., ... & 
Zettlemoyer, L. (2022). OPT: Open Pre-trained Transformer Language Models. 
arXiv preprint arXiv:2205.01068.
```

**Relevance:** Language model component in BLIP-2

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
# A prediction is correct if the recommended disease label matches test image label
recall_at_k = hits / total_test_samples
```

**Used Metrics:**
- **Recall@1**: Accuracy of top-1 recommendation
- **Recall@3**: Accuracy within top-3 recommendations
- **Recall@5**: Accuracy within top-5 recommendations

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

### 3. Cosine Similarity
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
similarity = torch.matmul(test_image_features, train_text_features.T)
# Features are L2-normalized, so this computes cosine similarity
```

---

## Technical Terminology

### Deep Learning Components

#### 1. **Feature Extraction**
**Definition:** The process of transforming raw data into numerical features that machine learning models can process.

**In this project:**
- **Image Features (1408-dimensional):** Visual representations from BLIP-2 vision encoder
- **Text Features (2560-dimensional):** Semantic representations from OPT language model

**Reference:** Bengio, Y., Courville, A., & Vincent, P. (2013). *Representation Learning: A Review and New Perspectives*. IEEE TPAMI, 35(8), 1798-1828.

---

#### 2. **Embeddings**
**Definition:** Dense vector representations of data (images or text) in a continuous vector space where similar items are positioned closer together.

**Properties:**
- Fixed-dimensional vectors
- Learned through neural network training
- Capture semantic meaning

**Reference:** Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality*. NeurIPS.

---

#### 3. **Normalization (L2 Normalization)**
**Definition:** Scaling vectors to unit length while preserving direction.

**Formula:**
```
normalized_vector = vector / ||vector||₂

where ||vector||₂ = sqrt(Σ(x_i²))
```

**Purpose:**
- Makes cosine similarity computation equivalent to dot product
- Removes magnitude bias, focuses on direction

**Implementation:**
```python
features = F.normalize(features, p=2, dim=-1)
```

---

#### 4. **Q-Former (Querying Transformer)**
**Definition:** BLIP-2's learnable query network that bridges frozen vision and language models.

**Architecture:**
- 32 learnable query embeddings
- Cross-attention to vision encoder
- Self-attention between queries
- Outputs vision-language aligned representations

**Reference:** Li et al. (2023) - BLIP-2 paper

---

#### 5. **Frozen Models**
**Definition:** Pre-trained models whose parameters are not updated during training on downstream tasks.

**Advantages:**
- Reduces computational cost
- Prevents catastrophic forgetting
- Enables efficient fine-tuning

**In this project:**
- Vision Encoder: Frozen ViT
- Language Model: Frozen OPT-2.7B
- Only Q-Former is trainable (in original BLIP-2 training)

---

### Data Processing Terms

#### 6. **Train-Test Split**
**Definition:** Dividing dataset into separate subsets for model training and evaluation.

**Configuration:**
- **Training Set (70%):** Used to build the retrieval database
- **Test Set (30%):** Used to evaluate recommendation performance
- **Stratified Split:** Maintains disease class distribution in both sets

**Reference:** Kohavi, R. (1995). *A Study of Cross-validation and Bootstrap for Accuracy Estimation and Model Selection*. IJCAI.

---

#### 7. **Batch Processing**
**Definition:** Processing multiple samples simultaneously to improve computational efficiency.

**Batch Sizes in this project:**
- Image feature extraction: 8 images/batch
- Text feature extraction: 16 texts/batch

**Benefits:**
- GPU parallelization
- Reduced processing time
- Memory efficiency

---

#### 8. **Stratified Sampling**
**Definition:** Sampling method that preserves the proportion of categories in subsets.

**Implementation:**
```python
train_test_split(stratify=skincon_df['label'])
```

**Purpose:** Ensures balanced representation of all skin diseases in train/test sets

---

### Retrieval System Terms

#### 9. **Information Retrieval (IR)**
**Definition:** Finding relevant information from large collections based on user queries.

**Components in this project:**
- **Query:** Test skin condition image
- **Database:** Training set herb-disease descriptions
- **Retrieval:** Finding top-K most similar herb treatments

**Reference:** Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.

---

#### 10. **Top-K Retrieval**
**Definition:** Selecting K items with highest similarity scores from a database.

**Implementation:**
```python
top_k_scores, top_k_indices = torch.topk(similarity_matrix, k=5)
```

**In this project:** K=5 (recommending 5 herb treatments per skin condition)

---

#### 11. **Cross-Modal Retrieval**
**Definition:** Finding relevant items from one modality (text) using queries from another modality (images).

**In this project:**
- **Query Modality:** Skin condition images
- **Database Modality:** Herbal treatment text descriptions
- **Bridge:** BLIP-2 vision-language model

**Reference:** Wang, K., Yin, Q., Wang, W., Wu, S., & Wang, L. (2016). *A Comprehensive Survey on Cross-modal Retrieval*. arXiv preprint arXiv:1607.06215.

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

**Version Used:** transformers >= 4.41.0
**Components Used:**
- `Blip2Processor`: Image and text preprocessing
- `Blip2Model`: Feature extraction model

---

### 3. Scikit-learn
**Reference:** Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825-2830.

**Citation:**
```
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., 
... & Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python. 
Journal of Machine Learning Research, 12(Oct), 2825-2830.
```

**Purpose:** Train-test splitting, evaluation metrics

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

## Methodology References

### 1. Image-Text Matching
**Reference:** Lee, K. H., Chen, X., Hua, G., Hu, H., & He, X. (2018). *Stacked Cross Attention for Image-Text Matching*. ECCV 2018.

**Approach:** Cross-modal similarity learning between visual and textual representations.

---

### 2. Feature Normalization for Similarity
**Reference:** Schroff, F., Kalenichenko, D., & Philbin, J. (2015). *FaceNet: A Unified Embedding for Face Recognition and Clustering*. CVPR 2015.

**Technique:** L2 normalization before similarity computation
**Benefit:** Makes cosine similarity equivalent to dot product

---

### 3. Zero-Shot Learning
**Reference:** Xian, Y., Lampert, C. H., Schiele, B., & Akata, Z. (2018). *Zero-Shot Learning—A Comprehensive Evaluation of the Good, the Bad and the Ugly*. IEEE TPAMI, 41(9), 2251-2265.

**Relevance:** BLIP-2's ability to generalize to unseen disease-herb combinations

---

### 4. Multi-Modal Learning
**Reference:** Baltrusaitis, T., Ahuja, C., & Morency, L. P. (2018). *Multimodal Machine Learning: A Survey and Taxonomy*. IEEE TPAMI, 41(2), 423-443.

**Citation:**
```
Baltrusaitis, T., Ahuja, C., & Morency, L. P. (2018). Multimodal Machine Learning: 
A Survey and Taxonomy. IEEE Transactions on Pattern Analysis and Machine Intelligence, 
41(2), 423-443.
```

**Relevance:** Foundation for vision-language integration in BLIP-2

---

## Additional Academic Context

### 1. Medical Image Analysis
**Reference:** Litjens, G., et al. (2017). *A Survey on Deep Learning in Medical Image Analysis*. Medical Image Analysis, 42, 60-88.

**Relevance:** Context for skin disease image analysis

---

### 2. Traditional Medicine Integration
**Reference:** Yuan, H., Ma, Q., Ye, L., & Piao, G. (2016). *The Traditional Medicine and Modern Medicine from Natural Products*. Molecules, 21(5), 559.

**Relevance:** Scientific basis for herbal medicine recommendations

---

### 3. Explainable AI in Healthcare
**Reference:** Holzinger, A., Biemann, C., Pattichis, C. S., & Kell, D. B. (2017). *What Do We Need to Build Explainable AI Systems for the Medical Domain?* arXiv preprint arXiv:1712.09923.

**Relevance:** Importance of interpretable recommendations in medical applications

---

## Project-Specific Methodology

### Data Processing Pipeline
1. **Data Loading:** CSV parsing with pandas
2. **Disease Matching:** Normalized string matching between datasets
3. **Left Join Strategy:** Preserving all test samples with intelligent fallback
4. **Train-Test Split:** 70/30 stratified split by disease label
5. **Feature Extraction:** Batch processing with BLIP-2
6. **Similarity Computation:** Cosine similarity via normalized dot product
7. **Top-K Retrieval:** PyTorch topk operation
8. **Evaluation:** Recall@K and MRR metrics

### Key Design Decisions

#### 1. LEFT Join vs INNER Join
**Decision:** Use LEFT join to preserve all skin condition images
**Rationale:** Ensures complete evaluation coverage even for diseases without exact herb matches
**Fallback Strategy:** Generic herbal treatments for unmatched diseases

#### 2. Feature Dimension Alignment
**Challenge:** Vision features (1408-dim) vs Text features (2560-dim) from BLIP-2 architecture
**Note:** Different dimensions are inherent to BLIP-2's design (ViT encoder vs OPT language model)

#### 3. Batch Size Selection
**Image Batch:** 8 (balanced GPU memory and processing speed)
**Text Batch:** 16 (text processing is less memory-intensive)

---

## Citation Template for This Project

```
[Your Name]. (2025). Herbal Treatment Recommendation System for Skin Diseases 
Using BLIP-2 Vision-Language Model. [Academic Institution]. 
Implementation based on Li et al. (2023) BLIP-2 architecture.
```

---

## Acknowledgments

This implementation builds upon:
- **BLIP-2 Model:** Salesforce Research
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
- **BLIP-2 Model Version:** Salesforce/blip2-opt-2.7b
- **Code Implementation:** Python 3.8+
- **Primary Framework:** PyTorch 2.x

---

## Contact for Academic Inquiries

For questions about this implementation or academic collaboration:
- Review the original BLIP-2 paper: arXiv:2301.12597
- Check Hugging Face model card: https://huggingface.co/Salesforce/blip2-opt-2.7b
- Consult course instructor or research advisor

---

*This document provides comprehensive academic references for the BLIP-2 Herbal Treatment Recommendation System project. All citations should be verified and updated according to your institution's citation guidelines.*
