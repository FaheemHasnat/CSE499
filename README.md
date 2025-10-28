# Skin Disease Image Datasets - DermNet & Skincon

This branch contains preprocessed image metadata datasets for skin disease classification and plant-based treatment recommendations.

## 📊 Datasets Overview

### 1. DermNet Dataset (`metadata_augmented.csv`)
- **Total Images:** 48,956
- **Disease Categories:** 23 broad categories
- **Source:** DermNet NZ - dermatology image library
- **Format:** CSV with image paths and class labels

**Categories Include:**
- Acne and Rosacea Photos
- Psoriasis Pictures Lichen Planus and related diseases
- Eczema Photos
- Melanoma Skin Cancer Nevi and Moles
- And 19 more skin condition categories

**Columns:**
- `image_path`: Path to the image file
- `class_name`: Disease category name
- Additional metadata fields

### 2. Skincon Dataset (`skincon_preprocessed.csv`)
- **Total Images:** 16,518
- **Specific Diseases:** 114 detailed disease classifications
- **Skin Tone Diversity:** Fitzpatrick scale 1-6
- **Morphological Features:** 50+ feature annotations
- **Source:** Skincon - diverse skin disease dataset

**Key Features:**
- Diverse skin tones (Fitzpatrick types 1-6)
- Detailed disease labels (e.g., `acne_vulgaris`, `psoriasis`, `melanoma`)
- Rich morphological annotations
- Comprehensive demographic metadata

**Columns:**
- `image_path`: Path to the image file
- `label`: Specific disease name
- `fitzpatrick`: Skin tone classification (1-6)
- 50+ morphological feature columns
- Demographic information

## 📁 File Structure

```
data/preprocessed/
├── metadata_augmented.csv      # DermNet metadata (48,956 images)
└── skincon_preprocessed.csv    # Skincon metadata (16,518 images)
```

## 🎯 Use Cases

### 1. **Skin Disease Classification**
- Train computer vision models to identify skin conditions
- 23 broad categories (DermNet) + 114 specific diseases (Skincon)
- Total: 65,474 images across diverse conditions

### 2. **Diverse Skin Tone Analysis**
- Skincon provides Fitzpatrick scale 1-6 annotations
- Ensures model fairness across different skin tones
- Critical for equitable healthcare AI

### 3. **Morphological Feature Learning**
- 50+ annotated features in Skincon
- Learn detailed visual characteristics of skin diseases
- Enhance diagnostic accuracy

### 4. **Plant-Based Treatment Recommendations**
- Integrate with HERB 2.0 database (disease-herb mappings)
- Provide plant-based treatment suggestions for identified conditions
- Vision Language Model (VLM) training for treatment recommendations

## 🔗 Integration with HERB 2.0

These datasets are designed to be integrated with the HERB 2.0 traditional medicine database:

1. **Disease Matching:** Map image dataset diseases to HERB 2.0 disease entities
2. **Treatment Extraction:** Link diseases to relevant herbs with skin treatment indications
3. **VLM Training:** Create vision-language pairs (image + treatment text)

**Related Files:** See main project repository for disease-herb mapping scripts and integration guides.

## 📊 Dataset Statistics

| Metric | DermNet | Skincon | Combined |
|--------|---------|---------|----------|
| Total Images | 48,956 | 16,518 | 65,474 |
| Disease Classes | 23 | 114 | 137 |
| Skin Tone Diversity | N/A | Fitzpatrick 1-6 | Partial |
| Morphological Features | Limited | 50+ | Variable |

## 🚀 Getting Started

### Load DermNet Data
```python
import pandas as pd

# Load DermNet metadata
dermnet = pd.read_csv('data/preprocessed/metadata_augmented.csv')

# View summary
print(f"Total images: {len(dermnet)}")
print(f"Disease categories: {dermnet['class_name'].nunique()}")
print(dermnet['class_name'].value_counts())
```

### Load Skincon Data
```python
import pandas as pd

# Load Skincon metadata
skincon = pd.read_csv('data/preprocessed/skincon_preprocessed.csv')

# View summary
print(f"Total images: {len(skincon)}")
print(f"Diseases: {skincon['label'].nunique()}")

# Check skin tone distribution
print(skincon['fitzpatrick'].value_counts().sort_index())
```

### Disease Distribution
```python
# Top 10 diseases in Skincon
print("\nTop 10 Most Common Diseases:")
print(skincon['label'].value_counts().head(10))

# Example output:
# psoriasis                          653
# squamous_cell_carcinoma           581
# lichen_planus                     491
# basal_cell_carcinoma              468
# allergic_contact_dermatitis       430
```

## 📝 Data Preprocessing

Both datasets have been preprocessed for machine learning:

✅ **Completed:**
- Image paths validated
- Missing values handled
- Label encoding prepared
- Duplicate entries removed
- Metadata standardized

✅ **Ready for:**
- Computer vision model training
- Disease classification tasks
- Treatment recommendation systems
- VLM fine-tuning

## 🔍 Dataset Quality

### DermNet
- ✅ High-quality clinical images
- ✅ Professionally curated
- ✅ Broad disease coverage
- ⚠️ Limited skin tone diversity

### Skincon
- ✅ Diverse skin tones (Fitzpatrick 1-6)
- ✅ Detailed annotations
- ✅ Specific disease labels
- ✅ Rich morphological features
- ✅ Demographic metadata

## 📚 References

### DermNet NZ
- **Website:** https://dermnetnz.org/
- **Description:** Comprehensive dermatology resource with clinical images
- **License:** Check DermNet website for usage terms

### Skincon Dataset
- **Paper:** [Insert paper citation if available]
- **Description:** Diverse skin disease dataset with Fitzpatrick annotations
- **License:** [Check dataset license]

## ⚠️ Important Notes

1. **Image Files Not Included:** This repository contains only metadata (CSV files). Image files must be downloaded separately from original sources.

2. **License Compliance:** Ensure you have proper licenses to use DermNet and Skincon images for your specific application.

3. **Medical Disclaimer:** These datasets are for research purposes only. Not intended for clinical diagnosis without expert medical supervision.

4. **Bias Awareness:** While Skincon includes diverse skin tones, always evaluate model performance across all demographic groups.

## 🛠️ Next Steps

1. **Download Images:** Obtain actual image files from DermNet and Skincon sources
2. **Disease Mapping:** Match diseases to HERB 2.0 database (see main project)
3. **Treatment Integration:** Link diseases to plant-based treatments
4. **Model Training:** Build classification or VLM models
5. **Evaluation:** Test on diverse skin tones and disease types

## 📧 Contact

For questions about this dataset branch:
- **Repository:** FaheemHasnat/CSE499
- **Branch:** dataset
- **Project:** VLM for Skin Disease Plant Treatment Recommendations

---

**Last Updated:** October 29, 2025  
**Dataset Version:** 1.0  
**Total Records:** 65,474 images
