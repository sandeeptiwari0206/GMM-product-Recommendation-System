<div align="center">

# 🤖 GMM Product Recommendation System

### Unsupervised Machine Learning — Customer Segmentation & Personalised Product Recommendations

[![Python](https://img.shields.io/badge/Python-99.8%25-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://github.com/sandeeptiwari0206/GMM-product-Recommendation-System)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-GMM-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com/sagemaker/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=for-the-badge)](https://www.apache.org/licenses/LICENSE-2.0)

<br/>

> *Cluster customers by behaviour using Gaussian Mixture Models and deliver personalised product recommendations at scale — locally or on AWS SageMaker.*

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [How GMM Works](#-how-gmm-works)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Tech Stack & Dependencies](#-tech-stack--dependencies)
- [Getting Started](#-getting-started)
- [Running the Pipeline](#-running-the-pipeline)
- [AWS SageMaker Deployment](#-aws-sagemaker-deployment)
- [Model Repacking](#-model-repacking-_repack_modelpy)
- [Testing](#-testing)
- [Output](#-output)
- [Author](#-author)

---

## 📖 Overview

This project implements a **production-grade, end-to-end product recommendation engine** using **Gaussian Mixture Models (GMM)** — a probabilistic unsupervised learning algorithm.

The system:
1. **Ingests** raw customer transaction data
2. **Preprocesses & engineers features** (purchase frequency, recency, category spend, etc.)
3. **Trains a GMM** to segment customers into behavioural clusters using soft probability assignments
4. **Profiles each cluster** to identify top products, spending patterns, and preferences
5. **Generates personalised recommendations** for every customer based on their most likely cluster
6. **Exports results** as CSV — ready for downstream use in marketing, CRM, or e-commerce platforms
7. **Deploys optionally** on AWS SageMaker for scalable, cloud-native inference

---

## 🧠 How GMM Works

> GMM is chosen over K-Means because it produces **soft cluster assignments** — each customer has a probability of belonging to every cluster, not just a hard label. This gives richer, more nuanced segmentation.

```
┌──────────────────────────────────────────────────────────┐
│                   Training Phase                          │
│                                                          │
│  Raw Data ──► Feature Engineering ──► Normalisation      │
│                                            │             │
│                                            ▼             │
│                               ┌─────────────────────┐   │
│                               │  GMM (EM Algorithm) │   │
│                               │                     │   │
│                               │  Fit N Gaussian     │   │
│                               │  distributions to   │   │
│                               │  customer vectors   │   │
│                               │                     │   │
│                               │  Select optimal N   │   │
│                               │  via BIC / AIC      │   │
│                               └──────────┬──────────┘   │
│                                          │               │
│                               Trained Model (joblib)     │
└──────────────────────────────────────────┼───────────────┘
                                           │
┌──────────────────────────────────────────▼───────────────┐
│                  Inference Phase                          │
│                                                          │
│  New Customer ──► Same Feature Pipeline                   │
│                          │                               │
│                          ▼                               │
│              GMM.predict_proba(customer)                  │
│                          │                               │
│         ┌────────────────┼──────────────────┐            │
│         ▼                ▼                  ▼            │
│    Cluster 0          Cluster 1  ...   Cluster N-1       │
│    P = 0.72           P = 0.18          P = 0.10         │
│         │                                                │
│         ▼                                                │
│   Top products from Cluster 0 → Recommendations         │
└──────────────────────────────────────────────────────────┘
```

---

## 🏗 Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                       Local / SageMaker                        │
│                                                               │
│  data/raw/          config/           src/                    │
│  ──────────         ──────────        ──────────────────────  │
│  raw CSVs     ──►   config.yaml  ──►  preprocessor.py         │
│                                       gmm_trainer.py          │
│                                       recommender.py          │
│                          │             │                      │
│                          ▼             ▼                      │
│                     scripts/        artifacts/                │
│                     ─────────       ──────────────────────    │
│                     train.py   ──►  gmm_model.joblib          │
│                     predict.py      scaler.joblib             │
│                     evaluate.py     cluster_profiles.json     │
│                          │                                    │
│                          ▼                                    │
│                       output/                                 │
│                       ──────────────────────────────────────  │
│                       all_customer_recommendations.csv        │
│                       cluster_summary.csv                     │
└───────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────▼───────────────┐
              │    AWS SageMaker (optional)    │
              │                               │
              │  Training Job ──► Model TAR   │
              │       │               │       │
              │  _repack_model.py ◄───┘       │
              │  (injects inference script)   │
              │       │                       │
              │  SageMaker Endpoint           │
              │  (real-time inference)        │
              └───────────────────────────────┘
```

---

## 📁 Project Structure

```
GMM-product-Recommendation-System/
│
├── src/                              # Core library — importable modules
│   ├── __init__.py
│   ├── preprocessor.py               # Feature engineering & normalisation
│   ├── gmm_trainer.py                # GMM model training, BIC/AIC selection
│   ├── recommender.py                # Cluster profiling & recommendation logic
│   └── utils.py                      # Shared helpers (logging, I/O, etc.)
│
├── scripts/                          # Runnable entry-point scripts
│   ├── train.py                      # End-to-end training pipeline
│   ├── predict.py                    # Generate recommendations for new data
│   └── evaluate.py                   # Model evaluation & cluster analysis
│
├── notebooks/                        # Jupyter notebooks for exploration & viz
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   ├── 02_gmm_tuning.ipynb           # Hyperparameter tuning (BIC/AIC curves)
│   └── 03_cluster_analysis.ipynb     # Cluster profiling & visualisation
│
├── config/
│   └── config.yaml                   # All tunable parameters (n_components,
│                                     # covariance_type, paths, S3 settings)
│
├── data/
│   └── raw/                          # Input CSV files (not committed)
│
├── artifacts/                        # Saved model files
│   ├── gmm_model.joblib              # Trained GMM model
│   ├── scaler.joblib                 # Feature scaler (StandardScaler)
│   └── cluster_profiles.json        # Top products & stats per cluster
│
├── output/                           # Generated recommendation CSVs
│   └── all_customer_recommendations.csv
│
├── tests/                            # Unit & integration tests (pytest)
│   ├── test_preprocessor.py
│   ├── test_gmm_trainer.py
│   └── test_recommender.py
│
├── docs/                             # Documentation & guides
│
├── _repack_model.py                  # AWS SageMaker model repacking utility
├── _repack_script_launcher.sh        # Shell launcher for repack job
├── GMM_Recommendation_System_Guide.docx  # Full project guide
├── GMM_Recommendation_System.zip    # Complete project archive
├── setup.py                          # Package installation config
└── requirements.txt                  # Python dependencies
```

---

## 🛠 Tech Stack & Dependencies

### Core ML
| Package | Version | Purpose |
|---------|---------|---------|
| `scikit-learn` | ≥ 1.2.0 | GMM model, preprocessing, evaluation |
| `numpy` | ≥ 1.23.0 | Numerical operations |
| `pandas` | ≥ 1.5.0 | Data loading & manipulation |
| `scipy` | ≥ 1.9.0 | Statistical functions |
| `joblib` | ≥ 1.2.0 | Model serialisation / deserialisation |

### Configuration
| Package | Version | Purpose |
|---------|---------|---------|
| `PyYAML` | ≥ 6.0 | Load `config/config.yaml` |

### AWS / Cloud (optional)
| Package | Version | Purpose |
|---------|---------|---------|
| `boto3` | ≥ 1.26.0 | AWS SDK — S3, SageMaker API |
| `sagemaker` | ≥ 2.140.0 | SageMaker Python SDK for pipelines & endpoints |

### Development
| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | ≥ 7.0.0 | Unit & integration testing |
| `pytest-cov` | ≥ 4.0.0 | Code coverage reports |
| `jupyter` | ≥ 1.0.0 | Notebooks for EDA & tuning |
| `matplotlib` | ≥ 3.6.0 | Visualisations |
| `seaborn` | ≥ 0.12.0 | Statistical plots |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip
- (Optional) AWS account with SageMaker & S3 access

### 1. Clone the Repository

```bash
git clone https://github.com/sandeeptiwari0206/GMM-product-Recommendation-System.git
cd GMM-product-Recommendation-System
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install as a Package (optional)

```bash
pip install -e .
```

### 4. Configure

Edit `config/config.yaml` to match your data paths and model settings:

```yaml
# config/config.yaml

data:
  raw_path: data/raw/customers.csv
  output_path: output/

model:
  n_components: 5          # number of GMM clusters
  covariance_type: full    # full | tied | diag | spherical
  max_iter: 200
  random_state: 42

recommendations:
  top_n_products: 10       # recommendations per customer

# AWS (only for SageMaker deployment)
aws:
  s3_bucket: your-bucket-name
  region: ap-south-1
```

---

## ▶️ Running the Pipeline

### Step 1 — Train the GMM Model

```bash
python scripts/train.py
```

This will:
- Load and preprocess data from `data/raw/`
- Fit a GMM with optimal `n_components` selected via BIC score
- Save `artifacts/gmm_model.joblib` and `artifacts/scaler.joblib`
- Export `artifacts/cluster_profiles.json`

### Step 2 — Generate Recommendations

```bash
python scripts/predict.py
```

This will:
- Load the saved model from `artifacts/`
- Assign each customer to their most probable cluster
- Output `output/all_customer_recommendations.csv`

### Step 3 — Evaluate & Analyse Clusters

```bash
python scripts/evaluate.py
```

Prints cluster statistics, silhouette scores, BIC/AIC curves, and top products per cluster.

### Explore in Notebooks

```bash
jupyter notebook notebooks/
```

---

## ☁️ AWS SageMaker Deployment

The system supports cloud deployment on AWS SageMaker for scalable batch or real-time inference.

### Training on SageMaker

```python
from sagemaker.sklearn.estimator import SKLearn

estimator = SKLearn(
    entry_point="scripts/train.py",
    framework_version="1.2-1",
    instance_type="ml.m5.large",
    role="arn:aws:iam::<account-id>:role/SageMakerRole",
    hyperparameters={
        "n-components": 5,
        "covariance-type": "full",
    }
)

estimator.fit({"training": "s3://your-bucket/data/raw/"})
```

### Deploying a Real-Time Endpoint

```python
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium"
)

# Predict cluster probabilities for a customer
response = predictor.predict(customer_features)
```

### Batch Transform (for all customers at once)

```python
transformer = estimator.transformer(
    instance_count=1,
    instance_type="ml.m5.large",
    output_path="s3://your-bucket/output/"
)

transformer.transform("s3://your-bucket/data/raw/customers.csv")
```

---

## 🔧 Model Repacking (`_repack_model.py`)

The `_repack_model.py` script is a **SageMaker utility** that repacks an existing trained model TAR archive with a custom inference entry point — without retraining.

This is used when you want to:
- Swap the inference script of an already-trained model
- Inject updated `code/inference.py` without re-running the full training job
- Add dependency scripts to an existing model archive

```bash
# Run via SageMaker Training Job or locally:
python _repack_model.py \
  --inference_script inference.py \
  --model_archive model.tar.gz \
  --source_dir src/
```

The shell launcher `_repack_script_launcher.sh` automates this as a SageMaker training step.

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# With coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_gmm_trainer.py -v
```

---

## 📂 Output

After running the pipeline, the following files are generated:

| File | Location | Description |
|------|----------|-------------|
| `gmm_model.joblib` | `artifacts/` | Serialised trained GMM model |
| `scaler.joblib` | `artifacts/` | Fitted StandardScaler for features |
| `cluster_profiles.json` | `artifacts/` | Top products & stats per cluster |
| `all_customer_recommendations.csv` | `output/` | Final recommendations for every customer |
| `cluster_summary.csv` | `output/` | Cluster membership counts & metrics |

### Sample Output Format

```
customer_id | cluster | probability | recommended_products
C001        | 2       | 0.89        | Product_A, Product_C, Product_F, ...
C002        | 0       | 0.76        | Product_B, Product_D, Product_G, ...
```

---

## 👨‍💻 Author

<div align="center">

**Sandeep Tiwari** — Cloud Engineer & DevOps Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sandeep-tiwari-616a33116/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/sandeeptiwari0206)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-3b82f6?style=flat-square)](https://your-portfolio-url.com)

📍 Jaipur, Rajasthan, India

</div>

---

<div align="center">

⭐ **If this project helped you, give it a star!**

</div>
