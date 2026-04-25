# Vision Language Model (VLM) for Predicting the Effects of Plant Compounds on Human Skin

## Project Overview
This project, **“Vision Language Model (VLM) for Predicting the Effects of Plant Compounds on Human Skin,”** aims to develop a multimodal AI system that understands and predicts the effects of medicinal plant compounds on various skin conditions.  

The model integrates both **visual data** (dermatological images) and **textual/biochemical data** (herbal compound descriptions) to establish meaningful correlations between plant-based treatments and their visual manifestations on human skin.  

The long-term goal is to enable **automated insights into the efficacy of natural compounds for dermatological applications** by leveraging modern **vision-language architectures**.

---

## Preprocessing and Dataset Preparation
So far, we have focused on preparing and preprocessing three major datasets — **SKINCON**, **DermNet**, and **HERB 2.0** — to make them compatible for multimodal training:

### **SKINCON**
- Multiple CSV files are merged into a single structured DataFrame linking image paths, metadata, and disease labels.  
- Redundant or missing entries are removed for dataset consistency.

### **DermNet**
- The dataset’s 23 skin condition folders are parsed and converted into labeled DataFrames.  
- Image paths are organized into training and testing sets for structured learning.

### **HERB 2.0**
- Three CSV files (`herb`, `ingredient`, and `disease` information) are read using robust parsing techniques.  
- Irrelevant identifiers and chemical metadata are dropped, and text fields are cleaned.  
- The files are merged to form a unified dataset highlighting the relationships between herbs, ingredients, and diseases.  
- A comprehensive `text_for_vlm` column is created to combine herb names, properties, and disease associations.

---

## Current Progress
The preprocessing pipeline successfully integrates **visual**, **textual**, and **biochemical** domains into a harmonized structure.  

Each image of a skin condition can now be **linked to descriptive herbal and compound information**, forming the foundation for multimodal model training.  

The next step is to:
- Generate **text embeddings** (using models like **BLIP-2**).  
- Align them with corresponding **skin disease images** for training the **Vision-Language Model (VLM)**.

---

*This project represents an interdisciplinary effort combining AI, dermatology, and phytochemistry to advance skin health research through multimodal learning.*
