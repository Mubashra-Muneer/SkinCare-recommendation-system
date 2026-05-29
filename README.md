# SkinMeta AI — Explainable AI-Based Personalized Skincare Recommendation System

SkinMeta AI is an Explainable AI-powered skincare recommendation system that combines Computer Vision, Machine Learning, NLP, and Rule-Based Dermatology Knowledge to analyze acne conditions from skin images and generate transparent, personalized skincare recommendations.

The system detects acne types using a CNN image classifier, maps predictions to dermatology-informed ingredients, filters unsafe products based on skin profile constraints, and recommends real skincare products with explainable reasoning.

---

# Features

* AI-powered acne classification from skin images
* Explainable skincare ingredient recommendations
* Real skincare product matching
* Hybrid recommendation engine with fallback ranking
* SHAP-based explainability
* Safety-aware ingredient filtering
* AM/PM skincare routine generation
* Real-time interactive Streamlit dashboard
* Performance evaluation with multiple metrics

---

# System Architecture

```text
Phase 1 → Phase 2 → Phase 3 → Web App

Skin Image
    ↓
CNN Classifier
    ↓
Dermatology Knowledge Base
    ↓
Product Bridge
    ↓
Safety Filtering
    ↓
Personalized Recommendations
    ↓
AM/PM Routine Generation
```

---

# Tech Stack

## Languages & Frameworks

* Python
* Streamlit

## Machine Learning & AI

* TensorFlow / Keras
* EfficientNetB0
* SHAP
* Scikit-learn

## Computer Vision

* OpenCV
* PIL

## NLP & Recommendation

* TF-IDF Vectorization
* Cosine Similarity
* Ridge Regression

## Data Processing

* Pandas
* NumPy

---

# Phase 1 — Data Acquisition & Preprocessing

## Dataset

* Kaggle: `tiswan14/acne-dataset-image`

## Preprocessing Pipeline

### Data Cleaning

* Corrupt image removal using PIL verification
* Exact duplicate removal using MD5 hashing
* Near-duplicate removal using perceptual hashing
* Dimension outlier filtering
* Missing value handling

### Image Processing

* Resize to `224×224`
* LANCZOS interpolation
* Normalization to `[0,1]`
* 9 augmentation strategies:

  * Rotation
  * Horizontal flip
  * Zoom
  * Brightness shift
  * Combined augmentations

### Dataset Split

* 70% Training
* 15% Validation
* 15% Testing

### Output

* `acne_dataset_manifest.csv`

---

# Phase 2A — Recommendation Data Preprocessing

## Dataset

* `dermatology_skincare.csv`

## Feature Engineering

Total engineered features: **159**

| Feature Group          | Count | Method                  |
| ---------------------- | ----- | ----------------------- |
| One-hot categorical    | 36    | Skin type, concern, age |
| Concentration features | 23    | StandardScaler          |
| Ingredients TF-IDF     | 50    | Bigram TF-IDF           |
| Effects TF-IDF         | 50    | Bigram TF-IDF           |

## Outputs

* `recommendation_features.csv`
* `recommendation_labels.csv`
* Pickled scalers/vectorizers

---

# Phase 2B — Product Bridge

The Product Bridge connects recommendation formulas to real skincare products.

## Product Filtering

* Detects ingredient columns automatically
* Filters facial skincare products only
* Identifies 22 acne-active ingredients

## Auto Acne-Type Tagging

Products are automatically tagged into:

* Comedonal
* Inflammatory
* Cystic
* Fungal
* Pigmentation
* Dark Circles

## TF-IDF Product Matrix

Features include:

* Product name
* Product type
* Brand
* Active ingredients
* Full ingredient lists
* Acne type mappings

Vocabulary size:

* Up to 2,000 features
* Unigrams + Bigrams

---

# Hybrid Recommendation Ranking

## Tier 1

Exact acne-type matching + ingredient overlap

## Tier 2

Family acne-type matching

## Tier 3

Universal safe fallback products

---

# Diversity Reranking

To improve recommendation quality:

* Same-brand penalty
* Ingredient-overlap penalty

---

# SHAP Explainability

A Ridge surrogate model with SHAP LinearExplainer explains:

* Why products were recommended
* Which ingredients increased similarity
* Feature importance strength:

  * Strong
  * Moderate
  * Small
  * Negligible

---

# Ingredient Conflict Detection

The system detects ingredient conflicts such as:

| Conflict                       | Severity |
| ------------------------------ | -------- |
| Benzoyl Peroxide + Retinol     | HIGH     |
| Benzoyl Peroxide + Vitamin C   | HIGH     |
| AHA/BHA + Retinol              | HIGH     |
| Multiple acids                 | MEDIUM   |
| Niacinamide + High % Vitamin C | LOW      |

---

# Phase 2C — Recommendation Engine

## 5-Stage Recommendation Pipeline

```text
Dermatology KB
    ↓
Hybrid Ranking
    ↓
Safety Filter
    ↓
SHAP Explainability
    ↓
AM/PM Routine Generator
```

## Dermatology Knowledge Base

Supports:

* 7 acne types
* 3 severity levels

### Acne Types

* Comedonal
* Inflammatory
* Cystic
* Fungal
* Pigmentation
* Dark Circles
* General

---

# Profile-Based Adjustments

## Sensitive Skin

Removes:

* Benzoyl Peroxide
* Retinol
* Glycolic Acid

Adds:

* Ceramides
* Centella
* Panthenol

## Dry Skin

Adds:

* Hyaluronic Acid
* Ceramides
* Squalane

## Oily Skin

Adds:

* Zinc PCA
* Niacinamide

## Age Rules

* `<20`: removes retinoids
* `>35`: adds anti-ageing retinol support

---

# Concentration Dosing

Ingredient concentrations scale by severity level.

Example:

| Ingredient       | Mild | Moderate | Severe |
| ---------------- | ---- | -------- | ------ |
| Salicylic Acid   | 0.5% | 1%       | 2%     |
| Benzoyl Peroxide | 2.5% | 5%       | 10%    |

---

# Safety Filtering

The filtering engine blocks:

* Skin-type incompatible products
* Severity-inappropriate actives
* Fragrance/alcohol for sensitive skin
* Pregnancy-unsafe ingredients

---

# AM/PM Routine Generator

Automatically builds:

* Morning skincare routines
* Night skincare routines

Features:

* Step ordering
* Night-only active detection
* SPF reminder generation

---

# Evaluation Metrics

The recommendation system is evaluated using:

* Ingredient Match Accuracy
* Acne-Type Relevance
* Safety Compliance Rate
* Diversity Score
* Overall Quality Score

---

# Phase 3 — CNN Image Classifier

## Architecture

* EfficientNetB0 backbone
* Custom classification head

## Training Strategy

### Phase A

* Frozen backbone
* Train dense head only

### Phase B

* Fine-tune last 30 layers
* Skip BatchNorm layers

---

# Acne Classes

| Class      | Severity | Type         |
| ---------- | -------- | ------------ |
| Blackheads | Mild     | Comedonal    |
| Whiteheads | Mild     | Comedonal    |
| Papules    | Moderate | Inflammatory |
| Pustules   | Moderate | Inflammatory |
| Cysts      | Severe   | Cystic       |

---

# Critical Preprocessing

Uses:

```python
preprocess_input()
```

instead of `/255` normalization to match EfficientNet preprocessing.

---

# Demo Mode

When no trained model exists, heuristic image analysis is used with:

* Texture analysis
* Redness detection
* Multi-scale feature extraction
* Temperature-scaled softmax

---

# Phase 4 — Streamlit Web Application

## UI Features

* Responsive dashboard
* Modern skincare-inspired color palette
* Card-based layouts
* Interactive results dashboard

## User Flow

```text
Landing Page
    ↓
Upload Image + Questionnaire
    ↓
CNN Analysis
    ↓
Recommendation Dashboard
```

---

# Dashboard Components

* CNN Prediction Results
* Confidence Visualization
* Ingredient Recommendations
* Product Recommendations
* SHAP Explanations
* Profile Adjustments
* AM/PM Routine Builder

---

# Testing & Evaluation

## Test Suite

* Online public-domain reference images
* Synthetic acne image generation

## Generated Outputs

* `test_results.csv`
* `confusion_matrix.png`
* `per_class_metrics.png`
* `confidence_distribution.png`
* `sample_predictions.png`
* `evaluation_report.txt`

---

# Key Technical Decisions

| Decision                | Reason                                  |
| ----------------------- | --------------------------------------- |
| EfficientNetB0          | Better accuracy-to-parameter ratio      |
| preprocess_input()      | Required preprocessing consistency      |
| Hybrid fallback ranking | Guarantees recommendations              |
| SHAP LinearExplainer    | Efficient sparse-feature explainability |
| Rule-based refinement   | Clinically interpretable logic          |
| Diversity reranking     | Prevents repetitive recommendations     |

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/skinmeta-ai.git
cd skinmeta-ai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

# Project Structure

```text
├── app.py
├── cnn_model.py
├── filters.py
├── skinmeta_recommendation_engine.py
├── step1b_product_bridge.py
├── phase2_skincare_data.py
├── test_cnn_model.py
├── datasets/
├── outputs/
├── models/
├── vectorizers/
├── scalers/
├── README.md
└── requirements.txt
```

---

# Future Improvements

* Multi-condition skin analysis
* Skin segmentation
* Mobile deployment
* Dermatologist feedback loop
* Retrieval-Augmented Generation (RAG)
* User progress tracking
* Real-time webcam support

---

# Disclaimer

SkinMeta AI is designed for educational and research purposes only and does not replace professional dermatological advice.

---

# Author

Mubashra Munir

---

If you found this project useful, consider giving it a star.
